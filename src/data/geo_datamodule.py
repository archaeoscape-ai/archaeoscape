from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components import (
    TDataset_gtiff,
    TiffSampler,
    VectorSampler,
    apply_split,
    compute_data_stat,
    prepare_target_profiles,
    set_data_path,
)
from src.utils import mkdir, pylogger
from src.utils.data import get_feats_with_proper_labels, gpkg_save
from src.utils.geoaffine import (
    Interpolation,
    itm_collate,
    sample_grid_squares_from_aoi_v2,
    sample_random_squares_from_aoi_v2,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SLRM:
    """Subtract local mean creating a Local Relief Model."""

    def __init__(self, mean_radius):
        self.mean_radius = mean_radius

    def __call__(self, Image):
        """Apply the transform on the image or list of images."""
        if isinstance(Image, List):
            New_Image = []
            for img in Image:
                mean = torch.nn.functional.avg_pool2d(
                    img,
                    kernel_size=self.mean_radius * 2 + 1,
                    stride=1,
                    padding=self.mean_radius,
                    count_include_pad=False,
                )
                New_Image.append(img - mean)
            return New_Image
        else:
            mean = torch.nn.functional.avg_pool2d(
                Image,
                kernel_size=self.mean_radius * 2 + 1,
                stride=1,
                padding=self.mean_radius,
                count_include_pad=False,
            )
            return Image - mean


class SLRM_min:
    """Subtract local min."""

    def __init__(self, mean_radius):
        self.mean_radius = mean_radius

    def __call__(self, Image):
        """Apply the transform on the image or list of images."""
        if isinstance(Image, List):
            New_Image = []
            for img in Image:
                mean = -torch.nn.functional.max_pool2d(
                    -img,
                    kernel_size=self.mean_radius * 2 + 1,
                    stride=1,
                    padding=self.mean_radius,
                )
                New_Image.append(img - mean)
            return New_Image
        else:
            mean = -torch.nn.functional.max_pool2d(
                -Image,
                kernel_size=self.mean_radius * 2 + 1,
                stride=1,
                padding=self.mean_radius,
            )
            return Image - mean


class GEODataModule(LightningDataModule):
    """`LightningDataModule`

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        aois_pth: str = "data/",
        layers_names: List[str] = "dtm",
        feats: str = "py@grab('../050_data/feats/gf_feats.gpkg')",
        labelkind: str = "easy3",
        subset_train: list = ["pkks_big_1m"],
        subset_val: list = ["pkks_small_1m"],
        subset_test: list = [],
        subset_pred: list = [],
        batch_size: int = 16,
        raster_targets: str = "",
        sample_multiplier: int = 1,
        imageside: int = 256,
        imagesize: int = 256,
        test_overlap: int = 0,
        mean: float = None,
        std: float = None,
        mean_type: str = "local",
        mean_radius: int = 40,
        num_workers: int = 0,
        iinter: Interpolation = 1,  # LINEAR
        pin_memory=True,
        tsize_base=None,
        tsize_enum_sizes=[1],
        tsize_enum_probs=None,
        tsize_range_frac=0,
        tsize_range_sizes=[0.5, 2],
        trot_angle=90,
        trot_prob=0.5,
        min_overlap=0.2,
        generate_targets=True,
    ) -> None:
        """Initialize a `LightningDataModule`.

        Args:
            aois_pth (str, optional): Path to aois. Defaults to "data/".
            layers_names (str, optional): Raster field type to use. Either dtm, ortho, ... Defaults to "dtm".
            feats (str, optional): Path to features. Defaults to "datasets/archeoscape/gf_features_merged.gpkg".
            labelkind (str, optional): Type of labels to use. Defaults to "easy3".
            subset_train (list, optional): Names of the area to use for training. Defaults to ["pkks_big_1m"].
            subset_val (list, optional): Names of the area to use for validation. Defaults to ["pkks_small_1m"].
            batch_size (int, optional): Batch size. Defaults to 16.
            sample_multiplier (int, optional): How many time an area should be sample (in esperance) per epoch. Defaults to 1.
            imageside (int, optional): Size of a sample in meters. Defaults to 256.
            imagesize (int, optional): Size of a sample in pixel. Defaults to 256.
            test_overlap (float, optional): Overlap for test samples. Defaults to 0.
            mean (float, optional): Mean for global normalization (if None computed). Defaults to None.
            std (float, optional): Standard deviation for normalization(if None computed). Defaults to None.
            mean_type (Union[str, List[str]], optional): Type of mean to use for normalization. Can be a list with a different value for each input channel. Either global, local, avg_pool or max_pool. Defaults to "local".
            mean_radius (int, optional): radius for neighbour based normalization(avg_pool or max_pool). Defaults to 40.
            num_workers (int, optional): Number of parrarel worker to load the datasets. Defaults to 0.
            iinter (Interpolation, optional): Interpolation type. Defaults to 1 ~ LINEAR.
            pin_memory (bool, optional): Use pin memory. Defaults to True.
            tsize_base (_type_, optional): For training : Default image side before augment.None for equal to image size dimension in meters. Defaults to None.
            tsize_enum_sizes (list, optional): For training : Randomly multiply the size by a factor in the sizeswith probs. Defaults to [1].
            tsize_enum_probs (_type_, optional): For training : Randomly multiply the size by a factor in the sizes with probs. Defaults to None.
            tsize_range_frac (int, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to 0.
            tsize_range_sizes (list, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to [0.5, 2].
            trot_angle (int, optional): For training : Randomly rotate. Defaults to 90.
            trot_prob (float, optional): For training : Randomly rotate. Defaults to 0.5.
            min_overlap (float, optional): For training : Minimum area of a sample that must be inside the raster. Defaults to 0.2.
        """

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.layers_names = layers_names
        if isinstance(self.layers_names, str):
            self.layers_names = [self.layers_names]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.epoch = 0

        self.rgen = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        log.info(f"Using {self.rgen=} for training dataset")

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except AttributeError:
            self.output_dir = "./logs"
        log.info(f"Using {self.output_dir=}")

        self.mean_radius = mean_radius

        # transform to be apply on gpu
        if mean_type == "avg_pool":
            self.normalizing_transform = SLRM(self.mean_radius)
        elif mean_type == "min_pool":
            self.normalizing_transform = SLRM_min(self.mean_radius)
        else:
            self.normalizing_transform = None

        self.generate_targets = generate_targets

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # fold_evalprep = mkdir(self.output_dir + "/eval_prepared")
        # # gf_aois, hshades = get_aois(cf['inputs.aois'])
        # gf_aois = gpd.read_file(self.hparams.aois_pth)
        # if "index" in gf_aois.columns:  # new dataset format
        #     gf_aois = gf_aois.set_index("index")
        #     datadir = Path(self.hparams.aois_pth).parent / "rasters"
        # elif "sblock" in gf_aois.columns:
        #     gf_aois = gf_aois.set_index("sblock")
        #     datadir = Path(self.hparams.aois_pth).parent / "rasters"
        # elif "aoi_name" in gf_aois.columns:  # old dataset format
        #     gf_aois = gf_aois.set_index("aoi_name")
        #     datadir = Path(self.hparams.aois_pth).parent
        # else:
        #     raise ("Unknown dataset format")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage(str): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        fold_evalprep = Path(self.output_dir + "/eval_prepared")

        gf_feats, label_names = get_feats_with_proper_labels(
            self.hparams.feats, self.hparams.labelkind
        )

        # To handle olderversion of the dataset
        gf_aois = gpd.read_file(self.hparams.aois_pth)
        if "index" in gf_aois.columns:  # new dataset format
            gf_aois = gf_aois.set_index("index")
            datadir = Path(self.hparams.aois_pth).parent / "rasters"
        elif "sblock" in gf_aois.columns:
            gf_aois = gf_aois.set_index("sblock")
            datadir = Path(self.hparams.aois_pth).parent / "rasters"
        elif "aoi_name" in gf_aois.columns:  # old dataset format
            gf_aois = gf_aois.set_index("aoi_name")
            datadir = Path(self.hparams.aois_pth).parent
        else:
            raise ("Unknown dataset format")

        # split the data
        gf_aois = apply_split(
            gf_aois,
            self.hparams.subset_train,
            self.hparams.subset_val,
            self.hparams.subset_test,
            self.hparams.subset_pred,
        )

        # set data path
        gf_aois = set_data_path(gf_aois, datadir, self.layers_names)

        # Assign number of samples to each gf_aois proportional to area (divisible by batchsize)
        batch_size = self.hparams.batch_size
        sample_multiplier = self.hparams.sample_multiplier
        imageside = self.hparams.imageside
        gf_aois["train_samples"] = (
            np.ceil(
                gf_aois.area / imageside**2 * sample_multiplier / batch_size
            ).astype(int)
            * batch_size
        )

        # raster targets profile
        raster_targets = prepare_target_profiles(gf_aois, self.layers_names[0])

        # Compute global mean and std
        WH = (self.hparams.imagesize, self.hparams.imagesize)  # in Pixel
        if self.hparams.mean is None or self.hparams.std is None:
            mean, std = compute_data_stat(
                gf_aois.query("split in ['train', 'val']"), self.layers_names
            )
        else:
            mean, std = self.hparams.mean, self.hparams.std
        log.info(f"global {mean=}, {std=}")

        # Construct gf_feats_dict for dataloader (dictionary of features clip to aoi)
        gf_feats_dict = {}
        for aoi_name, row_aoi in gf_aois.iterrows():
            poly_aoi = row_aoi["geometry"]
            gf_feats_dict[aoi_name] = gpd.clip(gf_feats, poly_aoi)

        iinter = Interpolation(self.hparams.iinter)

        # Create sampler for each aois
        self.samplers = {}
        for aoi_name, row_aoi in gf_aois.iterrows():
            # Create Sampler
            inputSamplers = []
            for layer in self.layers_names:
                inputSamplers.append(
                    TiffSampler(
                        row_aoi[layer],
                        aoi_name,
                        iinter,
                    )
                )
            targetSamplers = [
                VectorSampler(
                    gf_feats_dict[aoi_name],
                    label_names,
                    aoi_name,
                )
            ]

            self.samplers[aoi_name] = [
                inputSamplers,
                targetSamplers,
            ]

        # Instantiate reference grid-evaluating dataloaders for evaluation
        self.dloaders_gridsampling = {}
        for aoi_name, row_aoi in gf_aois.iterrows():
            poly_aoi = row_aoi["geometry"]
            raster_afft = raster_targets[aoi_name]["profile"]["transform"]
            if row_aoi["split"] == "train" or row_aoi["split"] == "val":
                eval_stride = self.hparams.imageside
                shift = (0, 0)
            elif row_aoi["split"] == "test" or row_aoi["split"] == "pred":
                eval_stride = self.hparams.imageside - int(
                    self.hparams.imageside * self.hparams.test_overlap
                )
                shift = (-eval_stride // 2, -eval_stride // 2)
            gf_squares = sample_grid_squares_from_aoi_v2(
                poly_aoi,
                imageside,
                gf_aois.crs,
                stride=eval_stride,
                shift=shift,
                raster_afft=raster_afft,
            )
            gf_squares["aoi_name"] = aoi_name
            gpkg_save(
                gf_squares,
                mkdir(fold_evalprep / "squares"),
                f"{aoi_name}_squares_r{self.trainer.global_rank}",
            )

            inputSamplers, targetSamplers = self.samplers[aoi_name]
            tdata_eval = TDataset_gtiff(
                gf_squares,
                inputSamplers,
                targetSamplers,
                mean,
                std,
                self.hparams.mean_type,
                WH=WH,
                generate_targets=self.generate_targets,
                return_debug_info=False,
            )
            tloader_eval = DataLoader(
                tdata_eval,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=itm_collate,
                drop_last=False,
                pin_memory=self.hparams.pin_memory,
                # persistent_workers=True,
            )
            self.dloaders_gridsampling[aoi_name] = tloader_eval

            # Check targets, assign number of squares inside
        gf_aois["eval_samples"] = {
            k: len(v.dataset.gf_squares)
            for k, v in self.dloaders_gridsampling.items()
        }

        # Print combined data stats
        for split, iids in gf_aois.groupby("split").groups.items():
            gf = gf_aois.loc[iids].drop(columns=self.layers_names)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                512,
            ):
                log.info(
                    "Split: {}, Area: {:.3f} km2, Samples train={} eval={}:\n{}".format(
                        split,
                        gf.area.sum() / 10**6,
                        gf.train_samples.sum(),
                        gf.eval_samples.sum(),
                        gf,
                    )
                )

        self.gf_aois = gf_aois
        self.gf_feats_dict = gf_feats_dict
        self.gf_feats = gf_feats
        self.raster_targets = raster_targets
        self.label_names = label_names
        self.mean = mean
        self.std = std
        self.iinter = iinter
        self.WH = WH

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        tsize_base = self.hparams.tsize_base
        if tsize_base is None:
            tsize_base = self.hparams.imageside
        tsize_enum_sizes = self.hparams.tsize_enum_sizes
        tsize_enum_probs = self.hparams.tsize_enum_probs
        tsize_range_frac = self.hparams.tsize_range_frac
        tsize_range_sizes = self.hparams.tsize_range_sizes
        trot_angle = self.hparams.trot_angle
        trot_prob = self.hparams.trot_prob
        min_overlap = self.hparams.min_overlap

        # Create squares dataset and appropriate dataset/dataloader
        rgen = self.rgen
        gf_train_squares = []
        train_datasets = []
        for aoi_name, row_aoi in self.gf_aois[
            self.gf_aois["split"] == "train"
        ].iterrows():
            poly_aoi = row_aoi["geometry"]
            raster_afft = self.raster_targets[aoi_name]["profile"]["transform"]
            n_samples = row_aoi["train_samples"]
            squares = sample_random_squares_from_aoi_v2(
                poly_aoi,
                n_samples,
                rgen,
                tsize_base,
                tsize_enum_sizes,
                tsize_enum_probs,
                tsize_range_frac,
                tsize_range_sizes,
                trot_angle,
                trot_prob,
                min_overlap,
                raster_afft,
            )
            gf_aoi_squares = gpd.GeoDataFrame(
                geometry=squares, crs=self.gf_aois.crs
            )
            gf_aoi_squares["aoi_name"] = aoi_name

            # create aoi dataloader
            inputSamplers, targetSamplers = self.samplers[aoi_name]
            tdata_train = TDataset_gtiff(
                gf_aoi_squares,
                inputSamplers,
                targetSamplers,
                self.mean,
                self.std,
                self.hparams.mean_type,
                WH=self.WH,
                generate_targets=self.generate_targets,
                return_debug_info=False,
            )
            train_datasets.append(tdata_train)

            gf_train_squares.append(gf_aoi_squares)

        gf_train_squares = pd.concat(
            gf_train_squares, axis=0, ignore_index=True
        )

        # Log for debugging
        dep_fold = (
            self.output_dir
            + f"/runtime/e{self.epoch}r{self.trainer.global_rank}"
        )
        gpkg_save(gf_train_squares, dep_fold, "train_squares")

        data_train = torch.utils.data.ConcatDataset(train_datasets)
        dload_train = DataLoader(
            data_train,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=True,  # Just a safeguard
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )

        self.epoch += 1
        return dload_train

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        validation_names = self.gf_aois.query("split == 'val'").index.tolist()
        train_names = (
            []
        )  # self.gf_aois.query("split == 'train'").index.tolist()
        valLoaders = [
            self.dloaders_gridsampling[name]
            for name in validation_names + train_names
        ]
        return valLoaders

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        test_names = self.gf_aois.query("split == 'test'").index.tolist()
        testLoaders = [self.dloaders_gridsampling[name] for name in test_names]
        return testLoaders

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        pred_names = self.gf_aois.query("split == 'pred'").index.tolist()
        predLoaders = [self.dloaders_gridsampling[name] for name in pred_names]
        return predLoaders

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Lightning hook that is called after data is moved to device. Used to normalize the data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]): The current batch of data, consisting of a tensor of data, a tensor of targets, and a dictionary of metadata.
            dataloader_idx (int): The index of the dataloader that produced this batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: The transformed batch, consisting of a tensor of normalized data, a tensor of targets, and a dictionary of metadata.
        """
        data, targets, metas = batch
        if self.normalizing_transform is not None:
            data = self.normalizing_transform(data)
        return data, targets, metas

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (Optional[str]): The stage being torn down. Either "fit", "validate", "test", or "predict".
                Defaults to None.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
                A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by `self.state_dict()`.

        Returns:
            None
        """
        pass


if __name__ == "__main__":
    _ = GEODataModule()
