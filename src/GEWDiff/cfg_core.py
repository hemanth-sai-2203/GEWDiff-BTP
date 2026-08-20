import torch
from tqdm.auto import tqdm

from .model.edm import ElucidatedDiffusion, default, exists


class CFGElucidatedDiffusion(ElucidatedDiffusion):
    """
    GEWDiff EDM with classifier-free conditioning dropout
    and classifier-free guidance during sampling.
    """

    def __init__(self, *args, p_drop=0.10, **kwargs):
        super().__init__(*args, **kwargs)

        if not 0.0 <= p_drop <= 1.0:
            raise ValueError("p_drop must be between 0 and 1")

        self.p_drop = p_drop

    def forward(self, img_lr, images, mask=None, edge=None):
        # 10% unconditional-conditioning dropout during training.
        if self.training and torch.rand((), device=images.device) < self.p_drop:
            img_lr = torch.zeros_like(img_lr)

            if mask is not None:
                mask = torch.zeros_like(mask)

            if edge is not None:
                edge = torch.zeros_like(edge)

        return super().forward(
            img_lr,
            images,
            mask=mask,
            edge=edge
        )

    @torch.no_grad()
    def sample_cfg(
        self,
        img_lr,
        batch_size=1,
        num_sample_steps=None,
        mask=None,
        guidance_scale=1.0,
        show_progress=True
    ):
        """
        CFG sampling.

        w = 0 : unconditional
        w = 1 : ordinary conditional prediction
        w > 1 : guided prediction

        D_cfg = D_uncond + w * (D_cond - D_uncond)
        """

        if guidance_scale < 0:
            raise ValueError("guidance_scale must be >= 0")

        device = img_lr.device

        num_sample_steps = default(
            num_sample_steps,
            self.num_sample_steps
        )

        sigmas = self.sample_schedule(
            num_sample_steps,
            device=device
        )

        shape = (
            batch_size,
            self.channels,
            self.image_size,
            self.image_size
        )

        images = sigmas[0] * torch.randn(
            shape,
            device=device
        )

        sigma_fn = lambda t: t.neg().exp()

        def t_fn(sigma):
            if isinstance(sigma, float):
                return torch.tensor(
                    sigma,
                    device=device
                ).log().neg()

            return sigma.log().neg()

        old_denoised = None

        iterator = tqdm(
            range(len(sigmas) - 1),
            disable=not show_progress
        )

        for i in iterator:

            sigma = sigmas[i].item()

            # Conditional prediction
            denoised_cond = self.preconditioned_network_forward(
                images,
                img_lr,
                sigma,
                mask,
                i=i
            )

            if guidance_scale == 1.0:

                denoised = denoised_cond

            else:

                # Unconditional conditioning
                zero_lr = torch.zeros_like(img_lr)

                zero_mask = (
                    torch.zeros_like(mask)
                    if mask is not None
                    else None
                )

                denoised_uncond = (
                    self.preconditioned_network_forward(
                        images,
                        zero_lr,
                        sigma,
                        zero_mask,
                        i=i
                    )
                )

                denoised = (
                    denoised_uncond
                    + guidance_scale
                    * (denoised_cond - denoised_uncond)
                )

            # Existing EDM second-order update
            t = t_fn(sigmas[i])
            t_next = t_fn(sigmas[i + 1])

            h = t_next - t

            if (
                not exists(old_denoised)
                or sigmas[i + 1] == 0
            ):
                denoised_d = denoised

            else:

                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h

                gamma = -1 / (2 * r)

                denoised_d = (
                    (1 - gamma) * denoised
                    + gamma * old_denoised
                )

            images = (
                (sigma_fn(t_next) / sigma_fn(t)) * images
                - (-h).expm1() * denoised_d
            )

            old_denoised = denoised

        return denoised, images
