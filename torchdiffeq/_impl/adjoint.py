import warnings
import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _mixed_norm, _all_callback_names, _all_adjoint_callback_names


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):
        # 在前向计算的过程中，将伴随灵敏度方法所需要的参数加入到ctx(contex)环境变量中来，可以在后续的backward中获取到相应的系统参数
        ctx.shapes = shapes
        ctx.func = func
        # todo: 存入环境参数中的变量的具体含义需要弄明白
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            # rtol:ode迭代终止的相对误差数值 atol:ode迭代终止的绝对误差数值
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
                ctx.save_for_backward(t, y, *adjoint_params)
            else:
                event_t, y = ans
                ctx.save_for_backward(t, y, event_t, *adjoint_params)

        return ans

    # 真正的核心在这里，主要是用于loss的反向梯度的计算，然后将计算好的梯度保存到对应的网络模组的梯度参数列表中，方便调用后续的梯度下降函数进行
    # 进行网络参数权重数值的更新（全文代码的核心，需要反复的理解）
    @staticmethod
    def backward(ctx, *grad_y): # 传入变量带星号表示对应的是一个变量列表类比于*args
        # pytorch的backward函数不支持在pycharm中进行断点，因此这里可以手动引入python的调试器
        # import pdb
        # pdb.set_trace()
        # grad_y->增强的初始状态delta(L)/delta(z(t1))
        with torch.no_grad():
            func = ctx.func
            # ode迭代相对误差
            adjoint_rtol = ctx.adjoint_rtol
            # ode迭代绝对误差
            adjoint_atol = ctx.adjoint_atol
            # 求解梯度的伴随方法
            adjoint_method = ctx.adjoint_method
            # 伴随方法的可选项
            adjoint_options = ctx.adjoint_options
            # t.requires_grad 也就是引入对于t的自动微分选项 一般来说都是True
            t_requires_grad = ctx.t_requires_grad

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            # 反向传播在ODE中等同于对于事件的时间进行积分操作
            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t, *adjoint_params = ctx.saved_tensors
                _t = t
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                grad_y = grad_y[1]
            else:
                t, y, *adjoint_params = ctx.saved_tensors
                grad_y = grad_y[0]

            # 可以获取到网络全部的结构参数信息并且转化为元组形态也由于后续的内部元素获取
            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################
            # torch.zeros_like:生成和括号内变量维度维度一致的全是零的
            # [-1] because y and grad_y are both of shape (len(t), *y0.shape) vjp_t就是损失函数关于t的梯度 vjp_y是损失函数关于隐藏层状态的梯度
            # import pdb
            # pdb.set_trace()
            # 增强的ODE状态初值设置
            aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params
            # 目前 aug_state = [vjp_t, y, vjp_y, vjp_params]
            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = y_aug[2] # 这里类比于论文中的a(t)
                # ignore gradients wrt time and parameters

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                    # wrt t here means we won't compute that if we don't need it.
                    # 论文当中的数值评价数值点
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    # torch.as_strided：此方法是根据现有tensor以及给定的步长来创建一个视图（类型仍然为tensor）
                    # 创建出来的b就是a的一个视图，可以发现，b中的元素都是a中的元素，所以其实b中并不存储数据，它只是显示a中的数据，如果改变a中的数据的话，b中的数据也会改变，反之亦然
                    _t = torch.as_strided(t, (), ())  # noqa
                    _y = torch.as_strided(y, (), ())  # noqa
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                    '''
                    torch.autograd.grad(
                        outputs,  # 计算图的数据结果张量--它就是需要进行求导的函数
                        inputs,  # 需要对计算图求导的张量--它是进行求导的变量
                        grad_outputs=None,  # 如果outputs不是标量，需要使用此参数 这里就是a(t) 作为雅各比矩阵的点乘输出项
                        retain_graph=None,  # 保留计算图
                        create_graph=None,  # 创建计算图
                        allow_unused=False  # inputs如果有不相关的变量，
                        # 即不对Outputs产生贡献的，比如你随便乱写一个inputs在那里，会报错，
                        # 改为True就不会报错，此时对这个垃圾inputs梯度为None返回。
                    )
                    '''
                    # import pdb # grad_outputs 是一个与因变量 y 的shape 一致的向量 对于返还回来的参数进行乘积操作
                    # pdb.set_trace() # t->1*1 y->(20,1,2)
                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                # zip打包成一个元组进行处理
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return (vjp_t, func_eval, vjp_y, *vjp_params)

            # Add adjoint callbacks
            for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
                try:
                    callback = getattr(func, adjoint_callback_name)
                except AttributeError:
                    pass
                else:
                    setattr(augmented_dynamics, callback_name, callback)

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            # 从后往前进行计算 通过伴随方法来替代传统神经网络反向传播的过程
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                # 这一步处理的意义在哪里:对于backward的伴随状态做一个梯度方向的修正
                aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            # Only compute gradient wrt initial time when in event handling mode.
            if event_mode and t_requires_grad:
                time_vjps = torch.cat([time_vjps[0].reshape(-1), torch.zeros_like(_t[1:])])

            adj_y = aug_state[2]
            adj_params = aug_state[3:]

        # backward函数的返回值，就是对应着forward里面的参数的梯度
        # def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
        #             adjoint_method,
        #             adjoint_options, t_requires_grad, *adjoint_params):
        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    # 引入对于odeSolve求解器中的关于迭代求解误差的定义
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):
    # 相当于做一次变量类型的检测
    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options['norm'] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == 'seminorm':
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm
