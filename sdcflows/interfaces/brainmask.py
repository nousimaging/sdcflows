# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Brain extraction interfaces."""
from nipype.interfaces.base import (
    traits,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    SimpleInterface,
)
from nipype.interfaces.freesurfer.base import FSTraitedSpecOpenMP, FSCommandOpenMP
from ..utils.tools import brain_masker
import os.path as op
import nibabel as nb
import numpy as np
import shutil

class _BrainExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="file to mask")


class _BrainExtractionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input file, after masking")
    out_mask = File(exists=True, desc="the binary brain mask")
    out_probseg = File(exists=True, desc="the probabilistic brain mask")


class BrainExtraction(SimpleInterface):
    """Brain extraction for EPI and GRE data."""

    input_spec = _BrainExtractionInputSpec
    output_spec = _BrainExtractionOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        (
            self._results["out_file"],
            self._results["out_probseg"],
            self._results["out_mask"],
        ) = brain_masker(
            self.inputs.in_file,
            fname_presuffix(self.inputs.in_file, suffix="_mask", newpath=runtime.cwd),
        )
        return runtime


class _BinaryDilationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="binary file to dilate")
    radius = traits.Float(3, usedefault=True, desc="structure element (ball) radius")


class _BinaryDilationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input file, after binary dilation")


class BinaryDilation(SimpleInterface):
    """Brain extraction for EPI and GRE data."""

    input_spec = _BinaryDilationInputSpec
    output_spec = _BinaryDilationOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _dilate(
            self.inputs.in_file,
            self.inputs.radius,
            newpath=runtime.cwd,
        )
        return runtime
    
def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    # Use mmap=False because we will be overwriting the output image
    resampled = nb.load(out_image, mmap=False)
    orig = nb.load(ref_image)

    if not np.allclose(orig.affine, resampled.affine):
        print(
            'Affines of input and reference images do not match, '
            'FMRIPREP will set the reference image headers. '
            'Please, check that the x-form matrices of the input dataset'
            'are correct and manually verify the alignment of results.')

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header['descrip'] = 'xform matrices modified by %s.' % (message or '(unknown)')

    newimg = resampled.__class__(resampled.get_fdata(), orig.affine, header)
    newimg.to_filename(out_image)

class CopyXFormInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')


class CopyXFormOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class CopyXForm(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """
    input_spec = CopyXFormInputSpec
    output_spec = CopyXFormOutputSpec

    def _run_interface(self, runtime):
        out_name = fname_presuffix(self.inputs.in_file,
                                   suffix='_xform',
                                   newpath=runtime.cwd)
        # Copy and replace header
        shutil.copy(self.inputs.in_file, out_name)
        _copyxform(self.inputs.hdr_file, out_name,
                   message='CopyXForm (niworkflows 0.5.2post5)')
        self._results['out_file'] = out_name
        return runtime
    
class _SynthStripInputSpec(FSTraitedSpecOpenMP):
    input_image = File(
        argstr="-i %s",
        exists=True,
        mandatory=True)
    no_csf = traits.Bool(
        argstr='--no-csf',
        desc="Exclude CSF from brain border.")
    border = traits.Int(
        argstr='-b %d',
        desc="Mask border threshold in mm. Default is 1.")
    gpu = traits.Bool(argstr="-g")
    out_brain = File(
        argstr="-o %s",
        name_template="%s_brain.nii.gz",
        name_source=["input_image"],
        keep_extension=False,
        desc="skull stripped image with corrupt sform")
    out_brain_mask = File(
        argstr="-m %s",
        name_template="%s_mask.nii.gz",
        name_source=["input_image"],
        keep_extension=False,
        desc="mask image with corrupt sform")


class _SynthStripOutputSpec(TraitedSpec):
    out_brain = File(exists=True)
    out_brain_mask = File(exists=True)


class SynthStrip(FSCommandOpenMP):
    input_spec = _SynthStripInputSpec
    output_spec = _SynthStripOutputSpec
    _cmd = "mri_synthstrip"

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update(
                {"OMP_NUM_THREADS": "1"}
            )


class FixHeaderSynthStrip(SynthStrip):

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderSynthStrip, self)._run_interface(
            runtime, correct_return_codes)

        outputs = self._list_outputs()
        if not op.exists(outputs["out_brain"]):
            raise Exception("mri_synthstrip failed!")

        if outputs.get("out_brain_mask"):
            _copyxform(
                self.inputs.input_image,
                outputs["out_brain_mask"])

        _copyxform(
            self.inputs.input_image,
            outputs["out_brain"])

        return runtime

class _UnionInputSpec(BaseInterfaceInputSpec):
    in1 = File(exists=True, mandatory=True, desc="binary file")
    in2 = File(exists=True, mandatory=True, desc="binary file")


class _UnionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input file, after binary dilation")


class Union(SimpleInterface):
    """Brain extraction for EPI and GRE data."""

    input_spec = _UnionInputSpec
    output_spec = _UnionOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _union(
            self.inputs.in1,
            self.inputs.in2,
            newpath=runtime.cwd,
        )
        return runtime


def _dilate(in_file, radius=3, newpath=None):
    """Dilate (binary) input mask."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from scipy import ndimage
    from skimage.morphology import ball
    from nipype.utils.filemanip import fname_presuffix

    mask = nb.load(in_file)
    newdata = ndimage.binary_dilation(
        np.asanyarray(mask.dataobj) > 0, ball(radius)
    )

    hdr = mask.header.copy()
    hdr.set_data_dtype("uint8")
    out_file = fname_presuffix(in_file, suffix="_dil", newpath=newpath or Path.cwd())
    mask.__class__(newdata.astype("uint8"), mask.affine, hdr).to_filename(
        out_file
    )
    return out_file


def _union(in1, in2, newpath=None):
    """Dilate (binary) input mask."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    mask = nb.load(in1)
    data = (
        np.asanyarray(mask.dataobj)
        + np.asanyarray(nb.load(in2).dataobj)
    ) > 0

    hdr = mask.header.copy()
    hdr.set_data_dtype("uint8")
    out_file = fname_presuffix(in1, suffix="_union", newpath=newpath or Path.cwd())
    mask.__class__(data.astype("uint8"), mask.affine, hdr).to_filename(
        out_file
    )
    return out_file
