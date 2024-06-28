from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import omegaconf
import pandas as pd
import rasterio
import shapely
import shapely.affinity
import torch

from src.utils.geoaffine import (
    convert_icell_to_tcell_v2,
    convert_vicell_to_vtcell,
    from_vecfeatures_determine_targets,
    sample_icell_from_raster_v2,
    sample_vicell_from_vector,
)


class Sampler:
    def __init__(
        self,
        aoi_name,
        factors_m=None,
        factors_px=None,
        mask_geometry=None,
    ):
        """Sampler virtual class for loading data from raster or vector sources.

        Args:
            aoi_name (str): name of the area of interest
            factors_m (List[int], optional): List of factors for up/downscaling in meters. 1 is included by default. Defaults to None.
            factors_px (List[int], optional): List of factors for up/downscaling in pixels. 1 is included by default. Defaults to None.
            mask_geometry (shapely.Geometry, optional): Mask geometry to enforce stricter mask. Defaults to None.
        """
        self.aoi_name = aoi_name
        self.mask_geometry = mask_geometry

        if factors_m is None:
            self.factors_m = [1]
        else:
            factors_m = list(factors_m)
            self.factors_m = [1] + factors_m
        if factors_px is None:
            self.factors_px = [1]
        else:
            factors_px = list(factors_px)
            self.factors_px = [1] + factors_px
        assert len(self.factors_m) == len(
            self.factors_px
        ), f"factors_m and factors_px have to be of same length {self.factors_m} != {self.factors_px}"

    def sample(
        self,
        geometry,
        side_px,
        safft_world_to_icell=None,
        safft_icell_to_tcell=None,
    ):
        """Sample data from the source at original scale.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels
            safft_world_to_icell (transform matrix, optional): matrix to convert from world to icell. Only necessary for VectorSampler. Defaults to None.
            safft_icell_to_tcell (transform matrix, optional): matrix to convert from icell to tcell. Only necessary for VectorSampler. Defaults to None.
        """
        return self.sample_single_scale(
            geometry, side_px, safft_world_to_icell, safft_icell_to_tcell
        )

    def sample_multi_scale(
        self,
        geometry,
        side_px,
        safft_world_to_icell=None,
        safft_icell_to_tcell=None,
    ):
        """Sample data from the source at multiple scales as defined by factors_m and factors_px.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels
            safft_world_to_icell (transform matrix, optional): matrix to convert from world to icell. Only necessary for VectorSampler. Defaults to None.
            safft_icell_to_tcell (transform matrix, optional): matrix to convert from icell to tcell. Only necessary for VectorSampler. Defaults to None.
        """
        images_nrm = []
        for i in range(len(self.factors_m) - 1, -1, -1):
            px_scaling_factor = self.factors_px[i]
            m_scaling_factor = self.factors_m[i]
            # increase square size by m_scaling_factor
            context_square = shapely.affinity.scale(
                geometry,
                xfact=m_scaling_factor,
                yfact=m_scaling_factor,
            )
            WH = side_px[0] * px_scaling_factor, side_px[1] * px_scaling_factor

            (
                I_nrm,
                safft_world_to_icell,
                safft_icell_to_tcell,
                window,
            ) = self.sample_single_scale(
                context_square,
                WH,
                safft_world_to_icell,
                safft_icell_to_tcell,
            )
            images_nrm.append(I_nrm)
        return (
            images_nrm[::-1],
            safft_world_to_icell,
            safft_icell_to_tcell,
            window,
        )

    def sample_single_scale(
        self,
        geometry,
        side_px,
        safft_world_to_icell,
        safft_icell_to_tcell,
        scaling=None,
    ):
        """Sample data from the source at a single scale. Needs to be implemented by the child class.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels
            safft_world_to_icell (transform matrix, optional): matrix to convert from world to icell. Only necessary for VectorSampler. Defaults to None.
            safft_icell_to_tcell (transform matrix, optional): matrix to convert from icell to tcell. Only necessary for VectorSampler. Defaults to None.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def get_mask(self, safft_world_to_icell, safft_icell_to_tcell, side_px):
        """Get mask for the area of interest.

        Args:
            safft_world_to_icell (transform matrix): matrix to convert from world to icell
            safft_icell_to_tcell (transform matrix): matrix to convert from icell to tcell
            side_px (Tuple[int, int]): Size of the square in pixels

        Returns:
            np.ndarray: Mask for the area of interest
        """

        mask = shapely.affinity.affine_transform(
            self.mask_geometry, safft_world_to_icell
        )
        mask = shapely.affinity.affine_transform(mask, safft_icell_to_tcell)
        raster_mask = rasterio.features.rasterize(
            shapes=[(mask, 0)],
            out_shape=side_px,
            fill=255,
            dtype=np.uint8,
            all_touched=False,
        )
        return raster_mask


class TiffSampler(Sampler):
    def __init__(
        self,
        raster_path,
        aoi_name,
        iinter,
        factors_m=None,
        factors_px=None,
        mask_geometry=None,
    ):
        super().__init__(
            aoi_name, factors_m, factors_px, mask_geometry=mask_geometry
        )
        self.raster_path = raster_path
        self.iinter = iinter

    def sample_single_scale(
        self,
        geometry,
        side_px,
        safft_world_to_icell=None,
        safft_icell_to_tcell=None,
    ):
        """Sample data from the source at a single scale. Needs to be implemented by the child class.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels
            safft_world_to_icell (transform matrix, optional): matrix to convert from world to icell. Not necessary for TiffSampler. Defaults to None.
            safft_icell_to_tcell (transform matrix, optional): matrix to convert from icell to tcell. Not necessary for TiffSampler. Defaults to None.
        """
        icell = sample_icell_from_raster_v2(self.raster_path, geometry)
        tcell = convert_icell_to_tcell_v2(
            icell["img"], icell["square"], side_px, self.iinter
        )
        if self.mask_geometry is not None:
            area_mask = self.get_mask(
                icell["safft_world_to_icell"],
                tcell["safft_icell_to_tcell"],
                side_px,
            )
            tcell["img"].mask = np.logical_or(
                tcell["img"].mask, area_mask[..., None]
            )
        return (
            tcell["img"],
            icell["safft_world_to_icell"],
            tcell["safft_icell_to_tcell"],
            icell["window"],
        )


class VectorSampler(Sampler):
    def __init__(
        self,
        gf_feats,
        label_names,
        aoi_name,
        factors_m=None,
        factors_px=None,
        mask_geometry=None,
    ):
        super().__init__(
            aoi_name, factors_m, factors_px, mask_geometry=mask_geometry
        )
        self.gf_feats = gf_feats
        self.label_names = label_names

    def sample_single_scale(
        self, geometry, side_px, safft_world_to_icell, safft_icell_to_tcell
    ):
        """Sample data from the source at a single scale. Needs to be implemented by the child class.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels
            safft_world_to_icell (transform matrix, optional): matrix to convert from world to icell. Necessary for VectorSampler. Defaults to None.
            safft_icell_to_tcell (transform matrix, optional): matrix to convert from icell to tcell. Necessary for VectorSampler. Defaults to None.
        """
        assert (
            safft_world_to_icell is not None
        ), "Vector sampler called before any raster. Need to sample raster first"
        vicell = sample_vicell_from_vector(
            self.gf_feats, geometry, safft_world_to_icell
        )
        vtcell = convert_vicell_to_vtcell(
            vicell["gf_cfeats"],
            vicell["square"],
            safft_icell_to_tcell=safft_icell_to_tcell,
        )
        targets = from_vecfeatures_determine_targets(
            vtcell["gf_cfeats"], self.label_names, side_px
        )
        return (
            targets["ssegm_mask"],
            safft_world_to_icell,
            safft_icell_to_tcell,
            None,
        )


class TDataset_gtiff(torch.utils.data.Dataset):
    def __init__(
        self,
        gf_squares: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
        inputSamplers: List[Sampler],
        targetSamplers: List[Sampler],
        mean: float,
        std: float,
        mean_type: str,
        WH: int = None,
        generate_targets=True,
        return_debug_info: bool = False,
    ):
        """Torch dataset for loading Tiff geodata.

        Args:
            gf_squares (gpd.GeoDataFrame): GeoDataFrame of squares to load
            InputSamplers (List[Sampler]): List of input samplers
            targetSamplers (List[Sampler]): List of target samplers
            mean (float): Mean for global normalization (if None computed)
            std (float): Standard deviation for normalization(if None computed)
            mean_type (str): Type of mean to use for normalization. Either global, local
            WH (Union[None, Tuple[int, int]], optional): Size of the image. Defaults to None.
            return_debug_info (bool, optional): Whether to return debug info. Defaults to False.
            generate_target (bool, optional): Whether to generate target. Defaults to True.
        """
        self.gf_squares = gf_squares
        self.aoi_name = inputSamplers[0].aoi_name
        self.inputSamplers = inputSamplers
        self.targetSamplers = targetSamplers
        for sampler in self.inputSamplers:
            assert sampler.aoi_name == self.aoi_name, "sampler AOI conflict"
        for sampler in self.targetSamplers:
            assert sampler.aoi_name == self.aoi_name, "sampler AOI conflict"

        self.mean_type = mean_type
        self.mean = mean
        self.std = std
        self.WH = WH
        self.return_debug_info = return_debug_info
        self.generate_targets = generate_targets

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.gf_squares)

    def sample_inputs(self, geometry, side_px):
        """Sample inputs from the inputSamplers.

        Args:
            geometry (shapely.Polygon): Square to sample
            side_px (Tuple[int, int]): Size of the square in pixels

        Returns:
            Tuple[List[torch.Tensor], np.ndarray, np.ndarray, Tuple[float, float, float, float]]: List of images, mask, safft_world_to_icell, safft_icell_to_tcell, window
        """

        safft_world_to_icell = None
        images_list = []
        window = None
        for sampler in self.inputSamplers:
            # sample at multiple scales from sampler
            (
                img_list,
                safft_world_to_icell,
                safft_icell_to_tcell,
                new_window,
            ) = sampler.sample_multi_scale(
                geometry, side_px, safft_world_to_icell
            )
            images_list.append([])
            # assume all images have the same shape HWC
            for img in img_list:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                images_list[-1].append(img)

            if new_window is not None:
                assert (
                    window is None or window == new_window
                ), "Window conflict"
                window = new_window

        # normalise all images
        I_nrm_list = []
        mask_list = []
        for scale in range(len(images_list[0])):
            # get all images at the same scale from all samplers and concatenate them
            images_scale = [
                images_list[k][scale] for k in range(len(images_list))
            ]
            image = np.concatenate(images_scale, axis=-1)
            mask = image.mask
            if mask is np.ma.nomask:
                mask = np.zeros_like(image, dtype=bool)
            mask_list.append(mask)

            # Normalize image. Missing values become exactly 0
            # remove global mean before conversion to tensor
            if isinstance(self.mean_type, str):
                if self.mean_type == "global":
                    mean = self.mean
                else:
                    mean = image.mean(axis=(0, 1))
            elif isinstance(self.mean_type, omegaconf.listconfig.ListConfig):
                mean_local = image.mean(axis=(0, 1))
                mean = [
                    (
                        mean_local[i]
                        if self.mean_type[i] == "local"
                        else self.mean[i]
                    )
                    for i in range(len(self.mean_type))
                ]
            else:
                raise ValueError(
                    f"Mean type not supported: {self.mean_type} of type {type(self.mean_type)}"
                )
            mean = np.asarray(mean)
            img_nrm = (image.filled(mean) - mean) / self.std

            I_nrm = torch.as_tensor(img_nrm).to(torch.float32)
            assert (
                I_nrm.ndim != 2
            ), f"Image has to be 3D, current shape{I_nrm.shape}"
            # convert from HWC to CHW
            I_nrm = I_nrm.permute(2, 0, 1)
            I_nrm_list.append(I_nrm)

        return (
            I_nrm_list,
            mask_list,
            safft_world_to_icell,
            safft_icell_to_tcell,
            window,
        )

    def sample_target(
        self, geometry, safft_world_to_icell, safft_icell_to_tcell, WH
    ):
        """Sample target from the targetSamplers.

        Args:
            geometry (shapely.Polygon): Square to sample
            safft_world_to_icell (np.ndarray): matrix to convert from world to icell
            safft_icell_to_tcell (np.ndarray): matrix to convert from icell to tcell
            WH (Tuple[int, int]): Size of the square in pixels

        Returns:
            torch.Tensor: Target tensor
        """
        targets = []
        for sampler in self.targetSamplers:
            # sample target from sampler (only a single scale is possible for target)
            (
                target,
                safft_world_to_icell,
                safft_icell_to_tcell,
                window,
            ) = sampler.sample(
                geometry, WH, safft_world_to_icell, safft_icell_to_tcell
            )
            if target.ndim == 2:
                target = np.expand_dims(target, axis=-1)
            targets.append(target)
        # concatenate all targets
        target = np.concatenate(targets, axis=-1)

        # convert from HWC to CHW
        target = torch.as_tensor(target)
        return target.permute(2, 0, 1).squeeze()

    def __getitem__(self, index):
        """Return the sample at the given index.

        Args:
            index (int): The index of the sample to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: A tuple containing the sample, the target, and the metadata.
        """
        row_square = self.gf_squares.iloc[index]

        (
            images_nrm,
            mask,
            safft_world_to_icell,
            safft_icell_to_tcell,
            window,
        ) = self.sample_inputs(
            row_square["geometry"],
            self.WH,
        )

        # Only keep mask for first (~ highest definition) image
        mask = mask[0]
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)
        if mask.sum() != 0:
            print(f"Mask sum is not empty for index {index}")
        meta = {
            "index": index,
            "aoi_name": self.aoi_name,
            "square": row_square["geometry"],
            "safft_world_to_icell": safft_world_to_icell,
            "safft_icell_to_tcell": safft_icell_to_tcell,
            "window": window,
            "mask": mask,
        }

        if self.generate_targets:
            targets = self.sample_target(
                row_square["geometry"],
                safft_world_to_icell,
                safft_icell_to_tcell,
                self.WH,
            )
            # set targets to NAN value if out of bounds
            mask = torch.as_tensor(mask)
            if (
                targets.dtype == torch.float32
                or targets.dtype == torch.float64
            ):
                targets[mask.expand_as(targets)] = torch.inf
            elif targets.dtype in [
                torch.int64,
                torch.int32,
            ]:
                targets[mask.expand_as(targets)] = 255
            else:
                raise ValueError(
                    f"Target dtype not supported: {targets.dtype}"
                )
        else:
            targets = torch.Tensor([])

        if len(images_nrm) == 1:
            return images_nrm[0], targets, meta
        else:
            return images_nrm, targets, meta
