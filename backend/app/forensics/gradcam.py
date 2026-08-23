import torch
class GradCAM:
    """Minimal gradient-CAM utility for a torch classifier and target convolutional layer."""
    def __init__(self,model,target_layer): self.model=model; self.target_layer=target_layer; self.activations=None; self.gradients=None; target_layer.register_forward_hook(self._forward); target_layer.register_full_backward_hook(self._backward)
    def _forward(self,module,inp,out): self.activations=out.detach()
    def _backward(self,module,gin,gout): self.gradients=gout[0].detach()
    def generate(self,x,target=None):
        self.model.zero_grad(); logits=self.model(x); idx=int(logits.argmax(1)) if target is None else target; logits[:,idx].sum().backward(); w=self.gradients.mean(dim=(2,3),keepdim=True); cam=(w*self.activations).sum(1).relu(); cam=cam/(cam.amax(dim=(1,2),keepdim=True)+1e-8); return cam.detach(),idx
