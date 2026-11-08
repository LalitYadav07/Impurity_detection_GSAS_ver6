"""Full-local three-curve attention; numerical weights use a distinct layout.

ELU+1 kernel attention uses matrix associativity (Katharopoulos et al., 2020,
https://arxiv.org/abs/2006.16236). It is not an approximation claim about the
older softmax encoder. Attention sees every convolution location before the
fixed average/maximum readout; at 4096 input bins that is 1024 locations.
"""
import math


def attend(q,k,v,valid=None):
    from torch.nn import functional as F
    q=F.elu(q)+1.;k=F.elu(k)+1.
    if valid is not None:k=k*valid[:,None,:,None]
    aggregate=k.transpose(-1,-2)@v
    denominator=(q*k.sum(-2,keepdim=True)).sum(-1,keepdim=True).clamp_min(1e-6)
    result=(q@aggregate)/denominator
    if valid is not None:result=result*valid[:,None,:,None]
    return result


def full_local(base,*,attention=True):
    import torch
    from torch import nn
    from .structural_attention_pool import adaptive

    class LinearBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm=nn.LayerNorm(32);self.qkv=nn.Linear(32,96,bias=False)
            self.output=nn.Linear(32,32);self.drop=nn.Dropout(.1)
            self.ff_norm=nn.LayerNorm(32)
            self.ff=nn.Sequential(nn.Linear(32,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,32))

        def forward(self,values,valid=None):
            batch,length,_=values.shape
            q,k,v=self.qkv(self.norm(values)).reshape(batch,length,3,4,8).permute(2,0,3,1,4)
            mapped=attend(q,k,v,valid).transpose(1,2).reshape(batch,length,32)
            values=values+self.drop(self.output(mapped))
            values=values+self.drop(self.ff(self.ff_norm(values)))
            if valid is not None:values=values.masked_fill(~valid[:,:,None],0.)
            return values

    class ReadoutPool(nn.Module):
        def forward(self,values):
            return .5*(adaptive(values,128)+adaptive(values,128,maximum=True))

    class FullLocal(nn.Module):
        def __init__(self):
            super().__init__();self.local=base.local;self.head=base.head
            self.position_scale=nn.Parameter(torch.tensor(.1))
            self.use_attention=attention;self.attention=LinearBlock() if attention else None
            self.amplitude=nn.Linear(3,64)
            nn.init.zeros_(self.amplitude.weight);nn.init.zeros_(self.amplitude.bias)
            self.pool=ReadoutPool()

        def embedding(self,values,coverage=None):
            """Shared 64-value readout before the final class projection.

            Keeping this inside the existing encoder lets conditional mode
            heads learn within-class changes without another curve engine.
            Existing state-dict names and the class forward path stay exact.
            """
            if values.ndim!=3 or values.shape[1]!=3:
                raise ValueError('Exactly three ordered curve channels required')
            if not 64<=values.shape[2]<=8192:
                raise ValueError('Full-local attention needs a bounded physical Q grid')
            valid=None
            if coverage is not None:
                if coverage.shape!=(values.shape[0],values.shape[2]):
                    raise ValueError('Coverage metadata must match the physical Q grid')
                if not torch.isfinite(coverage).all() or ((coverage<0)|(coverage>1)).any():
                    raise ValueError('Finite physical coverage between zero and one required')
                if not (coverage>0).any(1).all():
                    raise ValueError('An entirely masked pattern cannot be ranked')
                values=values.masked_fill((coverage==0)[:,None,:],0.)
            rms=values.square().mean(-1).sqrt()
            floor=rms[:,:2].amax(1,keepdim=True).clamp_min(1e-4)*1e-4
            local=self.local(values/torch.maximum(rms,floor).unsqueeze(-1))
            length=local.shape[-1]
            if coverage is not None:
                valid=adaptive(coverage[:,None,:],length,maximum=True).squeeze(1)>0
            if self.use_attention:
                position=torch.arange(length,device=values.device,dtype=values.dtype)[:,None]
                frequency=torch.exp(torch.arange(0,32,2,device=values.device,dtype=values.dtype)*(-math.log(10000.)/32))
                phase=position*frequency
                encoding=torch.stack((phase.sin(),phase.cos()),-1).reshape(length,32)
                encoded=self.attention(local.transpose(1,2)+self.position_scale*encoding,valid).transpose(1,2)
            else:
                encoded=local if valid is None else local.masked_fill(~valid[:,None,:],0.)
            pooled=self.pool(encoded)
            hidden=self.head[1](self.head[0](pooled))+self.amplitude(torch.log(rms.clamp_min(1e-8)))
            for layer in list(self.head.children())[2:-1]:hidden=layer(hidden)
            return hidden

        def forward(self,values,coverage=None):
            return self.head[-1](self.embedding(values,coverage))

    return FullLocal()
