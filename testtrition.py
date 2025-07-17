import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def test_triton_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available. Cannot test Triton.")

    print(f"Testing Triton kernel on {device}")
    size = 1024
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    output = torch.empty_like(x)

    grid = ((size + 255) // 256,)
    add_kernel[grid](
        x, y, output, size,
        BLOCK_SIZE=256
    )
    torch.cuda.synchronize()

    if torch.allclose(output, x + y, atol=1e-5):
        print("✅ Triton kernel ran successfully on GPU.")
    else:
        print("❌ Triton kernel output mismatch.")

if __name__ == "__main__":
    test_triton_gpu()