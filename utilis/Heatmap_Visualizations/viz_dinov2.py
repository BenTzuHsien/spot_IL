
import os, math, torch, cv2, imageio, numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from DinoMLP5_discretized import DinoMLP5_discretized



DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAJ_DIR = "/data/shared_data/SPOT_Real_World_Dataset/cleanup_dataset/" \
           "map01_01a/traj_000"

SCRIPT_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(SCRIPT_DIR, "Results"), exist_ok=True)
OUT_MP4    = os.path.join(SCRIPT_DIR, "Results", "dino_rollout.mp4")

IMG_SIZE     = (224, 224)
HEAD_FUSION  = "mean"
DISCARD      = 0.9
SELF_WEIGHT  = 0.2
FPS          = 1


tfs = transforms.Compose([
    ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# --------------- Attention Rollout Implementation sections --------------------
def get_all_selfattentions(vit, x):
    # Empty list to accumulate attention matrices for each Transformer Block
    attn_tensors = []
    
    def _hook(module, inp, _out):
        # N : number of tokens 
        # N = 261 = 256 + 4 (register tokens) + 1 [CLS]
        # C : embedding dimension (384)
        B, N, C = inp[0].shape
       
        # ---- Recomputing QKV projections ---
        # Linear projection layer calculates Q, K, V for us

        # shape is : [B, N , 3*C] concatenated Q, K, V
        # reshape result : [B, N, 3, num_heads, head_dim]
        # permute : [3, B, num_heads, N, head_dim]

        # number of attention heads : 6
        # per-head embedding dimension : 64
        qkv = module.qkv(inp[0]) \
                .reshape(B, N, 3, module.num_heads, C // module.num_heads) \
                .permute(2, 0, 3, 1, 4)
        

        # Q : [B, num_heads, N, head_dim]
        # K : [B, num_heads, N, head_dim]
        q, k = qkv[0], qkv[1]
        
        # attention calculation step
        # attention score between every pair of tokens
        # attn shape : [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        if getattr(module, "attn_bias", None) is not None:
            attn = attn + module.attn_bias

        # softmax step 
        attn_tensors.append(attn.softmax(-1).detach())

    handles = [blk.attn.register_forward_hook(_hook) for blk in vit.blocks]
    vit.eval()

    # Running a single forward pass on input  (triggers the hooks and fills attn_tensors)
    vit(x)  

    for h in handles:
        h.remove()

    # Return list of attention tensors for Attention Rollout
    # num_layers = 12
    # has 12 layers each w/ shape : [B, num_heads, N, N]
    return attn_tensors 


def compute_rollout(attn_list, head_fusion="max", discard_ratio=0.9,
                    self_weight=0.2, num_reg=4):
    
    # stacking attention matrices
    # attn: [L=12, B, head_dim, N, N]
    attn = torch.stack(attn_list)  

    # Head fusion
    # removing head_dim
    # attn: [L=12, B, N, N]
    if head_fusion == "mean":
        attn = attn.mean(dim=2)
    elif head_fusion == "max":
        attn = attn.max(dim=2).values
    elif head_fusion == "min":
        attn = attn.min(dim=2).values

    L, B, N, _ = attn.shape
    # Discard lowest attentions
    if discard_ratio > 0:
        flat = attn.flatten(2)
        thr = torch.quantile(flat, discard_ratio, dim=2).view(L, B, 1, 1)
        attn = torch.where(attn < thr, 0.0, attn)

    # Add scaled identity & normalize rows
    eye = torch.eye(N, device=attn.device)
    attn = attn + self_weight * eye.view(1, 1, N, N)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Attention Rollout
    rollout = eye.unsqueeze(0).repeat(B, 1, 1)
    for A in attn.flip(0): # from last layer to first
        # rollout shape : [B, N, N]
        rollout = A @ rollout

    # 0: CLS token index
    # 1 + num_reg = 5: start of patch tokens
    cls2patch = rollout[:, 0, 1 + num_reg:]
    # cls2patch shape : [B, 256]
    # 1 attention score per patch 
    return cls2patch
 # -------------------------------------------------------------------------


def load_five(folder):
    jpgs=sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])[:5]
    tens=[tfs(cv2.cvtColor(cv2.imread(os.path.join(folder,f)),cv2.COLOR_BGR2RGB)) for f in jpgs]
    return torch.stack(tens,0),jpgs

@torch.no_grad()
def main():
    model=DinoMLP5_discretized().to(DEVICE).eval()
    vit=(model.module if hasattr(model,"module") else model).shared_trunk
    n_reg=getattr(vit,"num_register_tokens",4)

    step_folders=sorted([d for d in os.listdir(TRAJ_DIR) if d.isdigit()])
    if not step_folders:
        print("Nothing to process"); return

    writer=imageio.get_writer(OUT_MP4, fps=FPS, codec='libx264', quality=8)

    for step in step_folders:
        step_dir=os.path.join(TRAJ_DIR,step)
        imgs_tensor,_=load_five(step_dir)
        if imgs_tensor.size(0)!=5:
            print(f"Skip {step_dir}"); continue

        imgs_tensor=imgs_tensor.to(DEVICE)                # (5,3,H,W)
        attn=get_all_selfattentions(vit,imgs_tensor)
        grid=int(math.sqrt(attn[0].shape[-1]-1-n_reg))
        roll=compute_rollout(attn,HEAD_FUSION,DISCARD,SELF_WEIGHT,n_reg)

        mean=torch.tensor([0.485,0.456,0.406],device=DEVICE).view(1,3,1,1)
        std =torch.tensor([0.229,0.224,0.225],device=DEVICE).view(1,3,1,1)
        rgb=(imgs_tensor*std+mean).clamp(0,1).permute(0,2,3,1).cpu().numpy()

        overlays, raws = [], []
        for i in range(5):
            heat=roll[i].reshape(grid,grid).cpu().numpy()
            heat=cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            heat=cv2.resize(heat,IMG_SIZE[::-1],cv2.INTER_CUBIC)
            heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
            img_bgr=cv2.cvtColor((rgb[i]*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
            over=cv2.addWeighted(img_bgr,0.6,heat,0.4,0)
            overlays.append(Image.fromarray(cv2.cvtColor(over,cv2.COLOR_BGR2RGB)))
            raws.append(Image.fromarray((rgb[i]*255).astype(np.uint8)))

        w,h=overlays[0].size
        canvas=Image.new("RGB",(w*5,h*2))
        order=[1,0,2,3,4]           
        for col,idx in enumerate(order):
            canvas.paste(overlays[idx],(col*w,0))   # top row
            canvas.paste(raws[idx],    (col*w,h))   # bottom row
        writer.append_data(np.array(canvas))

    writer.close()
    print(f"video saved : {OUT_MP4}")

if __name__=="__main__":
    main()
