import numpy as np
import scipy as sp
import scipy.signal
import skimage


class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    Moreover, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        input_grad = module.backward(input, output_grad)
    """
    def __init__(self):
        self._output = None
        self._input_grad = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad

    def _compute_output(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which will be stored in the `_output` field.

        Example: in case of identity operation:
        
        output = input 
        return output
        """
        raise NotImplementedError

    def _compute_input_grad(self, input, output_grad):
        """
        Returns the gradient of the module with respect to its own input. 
        The shape of the returned value is always the same as the shape of `input`.
        
        Example: in case of identity operation:
        input_grad = output_grad
        return input_grad
        """
        
        raise NotImplementedError
    
    def _update_parameters_grad(self, input, output_grad):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zero_grad(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def get_parameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def get_parameters_grad(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.

    def _compute_output(self, input):
        if self.training:
            output = (input - np.mean(input, axis=0)[np.newaxis, :]) / np.sqrt(np.var(input, axis=0)[np.newaxis, :] + BatchNormalization.EPS)
            self.moving_mean = self.moving_mean * self.alpha + np.mean(input, axis=0) * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + np.var(input, axis=0) * (1 - self.alpha)
        else:
            output = (input - self.moving_mean[np.newaxis, :]) / np.sqrt(self.moving_variance[np.newaxis, :] + BatchNormalization.EPS)
        return output

    def _compute_input_grad(self, input, output_grad):
        N,D = input.shape
        mu = np.mean(input, axis=0)
        xmu = input - mu[np.newaxis, :]
        sq = xmu ** 2
        var = 1./N * np.sum(sq, axis = 0)
        sqrtvar = np.sqrt(var + BatchNormalization.EPS)
        ivar = 1./sqrtvar
        xhat = xmu * ivar
        dxhat = output_grad
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. /(sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(var+BatchNormalization.EPS) * dsqrtvar
        dsq = 1. /N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1 + dx2
        if self.training:
            grad_input = dx
        return grad_input

    def __repr__(self):
        return "BatchNormalization"

class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output
        
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)
    
    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = []
        
    def _compute_output(self, input):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            output = input * self.mask / (1 - self.p)
        else:
            output = input
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.mask / (1 - self.p)
        return grad_input
        
    def __repr__(self):
        return "Dropout"


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def _compute_output(self, input):
        pad_size = self.kernel_size // 2

        batch_size, inp_chan, h, w = input.shape
        input_with_pad = np.zeros((batch_size, self.in_channels, h + 2 * pad_size, w + 2 * pad_size))

        for batch in range(batch_size):
            for chan in range(self.in_channels):
                input_with_pad[batch, chan] = np.pad(input[batch, chan], pad_width=pad_size)

        self._output = np.zeros((batch_size, self.out_channels, h, w))

        for batch in range(batch_size):
            for chan in range(self.out_channels):
                self._output[batch, chan] = scipy.signal.correlate(input_with_pad[batch], self.W[chan], mode='valid')[0] + self.b[chan]

        return self._output
    
    def _compute_input_grad(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        batch_size, out_chan, h, w = gradOutput.shape
        
        gradOutput_with_pad = np.zeros((batch_size, self.out_channels, h + 2 * pad_size, w + 2 * pad_size))
        for batch in range(batch_size):
            for chan in range(self.out_channels):
                gradOutput_with_pad[batch, chan] = np.pad(gradOutput[batch, chan], pad_width=pad_size)

        self._input_grad = np.zeros((batch_size, self.in_channels, h, w))

        W = np.rot90(self.W, k=2, axes=(2, 3))

        for batch in range(batch_size):
            for chan in range(self.in_channels):
                self._input_grad[batch, chan] = scipy.signal.correlate(gradOutput_with_pad[batch], W[:, chan, ...], mode='valid')[0]
        
        return self._input_grad
    
    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        batch_size, inp_chan, h, w = input.shape
        
        input_with_pad = np.zeros((batch_size, self.in_channels, h + 2 * pad_size, w + 2 * pad_size))
        for batch in range(batch_size):
            for chan in range(self.in_channels):
                input_with_pad[batch, chan] = np.pad(input[batch, chan], pad_width=pad_size)

        self.gradW = np.zeros_like(self.W)

        for batch in range(batch_size):
            for out_chan in range(self.out_channels):
                for in_chan in range(self.in_channels):
                    self.gradW[out_chan, in_chan] += scipy.signal.correlate(input_with_pad[batch, in_chan], gradOutput[batch, out_chan], mode='valid')

        self.gradb = np.zeros_like(self.b)

        for b in range(batch_size):
            self.gradb += np.sum(gradOutput[b], axis=(1, 2))

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
