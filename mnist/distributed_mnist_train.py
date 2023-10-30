from misc import *
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

rec_names = ["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter",
             "time_elapsed", "gamma", "corr"]
rec_unit = ["", "", "", "", "", "", "", "", "s", "min", "", ""]
import csv


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            lrscheduler: torch.optim.lr_scheduler.StepLR,
            gpu_id: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.lrscheduler = lrscheduler

    def _run_epoch(self, epoch, writer, itrcnt, loss_func, itr_arr, loss_arr, nfe_arr, time_arr, acc_arr,
                   forward_nfe_arr,
                   start_time, modelname, testnumber=0, evalfreq=1, lrscheduler=False, csvname='outdat.csv'):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        iter_start_time = time.time()
        acc = 0
        self.train_data.sampler.set_epoch(epoch)
        for x, y in tqdm(self.train_data):
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            itrcnt += 1
            self.model.module[1].df.nfe = 0
            self.optimizer.zero_grad()
            # forward in time and solve ode
            pred_y = self.model(x)
            if isinstance(pred_y, tuple):
                pred_y, rec = pred_y
                # compute loss
                loss = loss_func(pred_y, y) + 0.1 * torch.mean(rec)
            else:
                loss = loss_func(pred_y, y)
            forward_nfe_arr[epoch - 1] += self.model.module[1].df.nfe

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss.detach().cpu().numpy()
            nfe_arr[epoch - 1] += self.model.module[1].df.nfe
            acc += torch.sum((torch.argmax(pred_y, dim=1) == y).float())
        if self.lrscheduler:
            self.lrscheduler.step()
        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        forward_nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        backwardnfe = nfe_arr[epoch - 1] - forward_nfe_arr[epoch - 1]
        acc = acc.detach().cpu().numpy() / (len(self.train_data) * b_sz)
        if self.gpu_id == 0:
            try:
                gamma = self.model.module[1].df.gamma.val.detach().cpu().numpy()[0]
            except BaseException:
                gamma = 0
            try:
                corr = self.model.module[1].df.corr.val.detach().cpu().numpy()[0]
            except BaseException:
                corr = 0
            printouts = [modelname, testnumber, 'train', epoch,
                         loss_arr[epoch - 1], acc, forward_nfe_arr[epoch - 1],
                         backwardnfe, time_arr[epoch - 1],
                         (time.time() - start_time) / 60,
                         gamma, corr]

            print(str_rec(rec_names, printouts, rec_unit))
            writer.writerow(printouts)

            # 在gpu:0中单独进行mnist测试集中进行test，并且将测试结果记录到对应data文件中
            if epoch % evalfreq == 0:
                self.model.module[1].df.nfe = 0
                test_time = time.time()
                loss = 0
                acc = 0
                bcnt = 0
                for x, y in self.test_data:
                    # forward in time and solve ode
                    x = x.to(device=self.gpu_id)
                    y = y.to(device=self.gpu_id)
                    pred_y = self.model(x)
                    if isinstance(pred_y, tuple):
                        pred_y, rec = pred_y
                    pred_l = torch.argmax(pred_y, dim=1)
                    acc += torch.sum((pred_l == y).float())
                    bcnt += 1
                    # compute loss
                    loss += loss_func(pred_y, y) * y.shape[0]
                test_time = time.time() - test_time
                loss = loss.detach().cpu().numpy() / 10000
                acc = acc.detach().cpu().numpy() / 10000
                printouts = [modelname, testnumber, 'test', epoch,
                             loss, acc, self.model.module[1].df.nfe / len(self.test_data),
                             0, test_time,
                             (time.time() - start_time) / 60, 0, 0]
                print(str_rec(rec_names, printouts, rec_unit))
                writer.writerow(printouts)
                acc_arr[epoch - 1] = acc

    def _save_checkpoint(self, epoch):
        #  主要进行GPU的位置映射，防止只有主GPU在运行训练进程
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs, modelname, testnumber=0, evalfreq=1, lrscheduler=True,
              csvname='outdat.csv'):
        itrcnt = 0
        loss_func = nn.CrossEntropyLoss().cuda(self.gpu_id)
        itr_arr = np.zeros(max_epochs)
        loss_arr = np.zeros(max_epochs)
        nfe_arr = np.zeros(max_epochs)
        time_arr = np.zeros(max_epochs)
        acc_arr = np.zeros(max_epochs)
        forward_nfe_arr = np.zeros(max_epochs)

        # 构建csv文件的输出流
        csvfile = open(csvname, 'a+')
        writer = csv.writer(csvfile)

        start_time = time.time()
        for epoch in range(max_epochs):
            self._run_epoch(epoch, writer=writer, itrcnt=itrcnt, loss_func=loss_func, itr_arr=itr_arr,
                            loss_arr=loss_arr,
                            nfe_arr=nfe_arr, time_arr=time_arr, acc_arr=acc_arr, forward_nfe_arr=forward_nfe_arr,
                            start_time=start_time,
                            modelname=modelname, testnumber=testnumber, evalfreq=evalfreq, lrscheduler=lrscheduler,
                            csvname=csvname)
            if self.gpu_id == 0:
                self._save_checkpoint(epoch)

        csvfile.close()
