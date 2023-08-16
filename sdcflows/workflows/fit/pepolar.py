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
"""Datasets with multiple phase encoded directions."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import data
from ...interfaces.brainmask import FixHeaderSynthStrip, CopyXForm

INPUT_FIELDS = ("metadata", "in_data")
_PEPOLAR_DESC = """\
A *B<sub>0</sub>*-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references """

def init_3dQwarp_wf(omp_nthreads=1, debug=False, sloppy=False, name="pepolar_estimate_wf"):
    """
    Create the PEPOLAR field estimation workflow based on AFNI's ``3dQwarp``.

    This workflow takes in two EPI files that MUST have opposed
    :abbr:`PE (phase-encoding)` direction.
    Therefore, EPIs with orthogonal PE directions are not supported.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_3dQwarp_wf
            wf = init_3dQwarp_wf()

    Parameters
    ----------
    debug : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    in_data : :obj:`list` of :obj:`str`
        A list of two EPI files, the first of which will be taken as reference.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of the first element of ``in_data``.

    """
    from nipype.interfaces import afni
    from niworkflows.interfaces.header import CopyHeader
    from niworkflows.interfaces.fixes import (
        FixHeaderRegistration as Registration,
        FixHeaderApplyTransforms as ApplyTransforms,
    )
    from niworkflows.interfaces.freesurfer import StructuralReference
    from ...utils.misc import front as _front, last as _last
    from ...interfaces.utils import Flatten, ConvertWarp

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""{_PEPOLAR_DESC} \
with `3dQwarp` (@afni; AFNI {''.join(['%02d' % v for v in afni.Info().version() or []])}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_data", "metadata"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "fmap_ref"]), name="outputnode"
    )

    flatten = pe.Node(Flatten(), name="flatten")
    sort_pe = pe.Node(
        niu.Function(function=_sorted_pe, output_names=["sorted", "qwarp_args"]),
        name="sort_pe",
        run_without_submitting=True,
    )

    merge_pes = pe.MapNode(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            fixed_timepoint=True,  # Align to first image
            intensity_scaling=True,
            # 7-DOF (rigid + intensity)
            no_iteration=True,
            subsample_threshold=200,
            out_file="template.nii.gz",
        ),
        name="merge_pes",
        iterfield=["in_files"],
    )

    pe0_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name="pe0_wf"
    )
    pe1_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name="pe1_wf"
    )

    align_pes = pe.Node(
        Registration(
            from_file=data.load("translation_rigid.json"),
            output_warped_image=True,
        ),
        name="align_pes",
        n_procs=omp_nthreads,
    )

    qwarp = pe.Node(
        afni.QwarpPlusMinus(
            blur=[-1, -1],
            environ={"OMP_NUM_THREADS": f"{min(omp_nthreads, 4)}"},
            minpatch=9,
            nopadWARP=True,
            noweight=True,
            pblur=[0.05, 0.05],
        ),
        name="qwarp",
        n_procs=min(omp_nthreads, 4),
    )

    to_ants = pe.Node(ConvertWarp(), name="to_ants", mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name="cphdr_warp", mem_gb=0.01)

    unwarp_reference = pe.Node(
        ApplyTransforms(
            dimension=3,
            float=True,
            interpolation="LanczosWindowedSinc",
        ),
        name="unwarp_reference",
    )

    # fmt: off
    workflow.connect([
        (inputnode, flatten, [("in_data", "in_data"),
                              ("metadata", "in_meta")]),
        (flatten, sort_pe, [("out_list", "inlist")]),
        (sort_pe, qwarp, [("qwarp_args", "args")]),
        (sort_pe, merge_pes, [("sorted", "in_files")]),
        (merge_pes, pe0_wf, [(("out_file", _front), "inputnode.in_file")]),
        (merge_pes, pe1_wf, [(("out_file", _last), "inputnode.in_file")]),
        (pe0_wf, align_pes, [("outputnode.skull_stripped_file", "fixed_image")]),
        (pe1_wf, align_pes, [("outputnode.skull_stripped_file", "moving_image")]),
        (pe0_wf, qwarp, [("outputnode.skull_stripped_file", "in_file")]),
        (align_pes, qwarp, [("warped_image", "base_file")]),
        (inputnode, cphdr_warp, [(("in_data", _front), "hdr_file")]),
        (qwarp, cphdr_warp, [("source_warp", "in_file")]),
        (cphdr_warp, to_ants, [("out_file", "in_file")]),
        (to_ants, unwarp_reference, [("out_file", "transforms")]),
        (inputnode, unwarp_reference, [("in_reference", "reference_image"),
                                       ("in_reference", "input_image")]),
        (unwarp_reference, outputnode, [("output_image", "fmap_ref")]),
        (to_ants, outputnode, [("out_file", "fmap")]),
    ])
    # fmt: on
    return workflow


def _sorted_pe(inlist):
    """
    Generate suitable inputs to ``3dQwarp``.

    Example
    -------
    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-AP_bold.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-PA_bold.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-AP_sbref.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_sbref.nii.gz", {"PhaseEncodingDirection": "j"}),
    ... ])
    >>> paths[0]
    ['dir-AP_epi.nii.gz', 'dir-AP_bold.nii.gz', 'dir-AP_sbref.nii.gz']

    >>> paths[1]
    ['dir-PA_epi.nii.gz', 'dir-PA_bold.nii.gz', 'dir-PA_sbref.nii.gz']

    >>> args
    '-noXdis -noZdis'

    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-LR_epi.nii.gz", {"PhaseEncodingDirection": "i"}),
    ... ])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    """
    out_ref = [inlist[0][0]]
    out_opp = []

    ref_pe = inlist[0][1]["PhaseEncodingDirection"]
    for d, m in inlist[1:]:
        pe = m["PhaseEncodingDirection"]
        if pe == ref_pe:
            out_ref.append(d)
        elif pe[0] == ref_pe[0]:
            out_opp.append(d)
        else:
            raise ValueError("Cannot handle orthogonal PE encodings.")

    return (
        [out_ref, out_opp],
        {"i": "-noYdis -noZdis", "j": "-noXdis -noZdis", "k": "-noXdis -noYdis"}[
            ref_pe[0]
        ],
    )

def init_enhance_and_skullstrip_bold_wf(
    brainmask_thresh=0.5,
    name="enhance_and_skullstrip_bold_wf",
    omp_nthreads=1,
    pre_mask=False,
):
    """
    Pulled from niworkflows. FSL removed
    Enhance and run brain extraction on a BOLD EPI image.

    This workflow takes in a :abbr:`BOLD (blood-oxygen level-dependant)`
    :abbr:`fMRI (functional MRI)` average/summary (e.g., a reference image
    averaging non-steady-state timepoints), and sharpens the histogram
    with the application of the N4 algorithm for removing the
    :abbr:`INU (intensity non-uniformity)` bias field and calculates a signal
    mask.

    Steps of this workflow are:

      1. Calculate a tentative mask by registering (9-parameters) to *fMRIPrep*'s
         :abbr:`EPI (echo-planar imaging)` -*boldref* template, which
         is in MNI space.
         The tentative mask is obtained by resampling the MNI template's
         brainmask into *boldref*-space.
      2. Binary dilation of the tentative mask with a sphere of 3mm diameter.
      3. Run ANTs' ``N4BiasFieldCorrection`` on the input
         :abbr:`BOLD (blood-oxygen level-dependant)` average, using the
         mask generated in 1) instead of the internal Otsu thresholding.
      4. Skullstrip with freesurfer's synthstrip
      5. Apply final mask on the enhanced reference.

    Step 1 can be skipped if the ``pre_mask`` argument is set to ``True`` and
    a tentative mask is passed in to the workflow through the ``pre_mask``
    Nipype input.


    Workflow graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
            wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=1)

    .. _N4BiasFieldCorrection: https://hdl.handle.net/10380/3053

    Parameters
    ----------
    brainmask_thresh: :obj:`float`
        Lower threshold for the probabilistic brainmask to obtain
        the final binary mask (default: 0.5).
    name : str
        Name of workflow (default: ``enhance_and_skullstrip_bold_wf``)
    omp_nthreads : int
        number of threads available to parallel nodes
    pre_mask : bool
        Indicates whether the ``pre_mask`` input will be set (and thus, step 1
        should be skipped).

    Inputs
    ------
    in_file : str
        BOLD image (single volume)
    pre_mask : bool
        A tentative brain mask to initialize the workflow (requires ``pre_mask``
        parameter set ``True``).


    Outputs
    -------
    bias_corrected_file : str
        the ``in_file`` after `N4BiasFieldCorrection`_
    skull_stripped_file : str
        the ``bias_corrected_file`` after skull-stripping
    mask_file : str
        mask of the skull-stripped input file
    out_report : str
        reportlet for the skull-stripping

    """
    from niworkflows.interfaces.nibabel import ApplyMask, BinaryDilation
    from niworkflows.interfaces.fixes import FixN4BiasFieldCorrection as N4BiasFieldCorrection

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_file", "pre_mask"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["mask_file", "skull_stripped_file", "bias_corrected_file"]
        ),
        name="outputnode",
    )

    # Run N4 normally, force num_threads=1 for stability (images are small, no need for >1)
    n4_correct = pe.Node(
        N4BiasFieldCorrection(
            dimension=3, copy_header=True, bspline_fitting_distance=200
        ),
        shrink_factor=2,
        name="n4_correct",
        n_procs=1,
    )
    n4_correct.inputs.rescale_intensities = True

    # Create a generous BET mask out of the bias-corrected EPI
    synthstrip = pe.Node(FixHeaderSynthStrip(), name = "synthstrip")

    # Use AFNI's unifize for T2 contrast & fix header
    #unifize = pe.Node(
        #afni.Unifize(
            #t2=True,
            #outputtype="NIFTI_GZ",
            # Default -clfrac is 0.1, 0.4 was too conservative
            # -rbt because I'm a Jedi AFNI Master (see 3dUnifize's documentation)
            #args="-clfrac 0.2 -rbt 18.3 65.0 90.0",
            #out_file="uni.nii.gz",
        #),
        #name="unifize",
    #)
    #fixhdr_unifize = pe.Node(CopyXForm(), name="fixhdr_unifize", mem_gb=0.1)

    #fixhdr_skullstrip = pe.Node(CopyXForm(), name="fixhdr_skullstrip", mem_gb=0.1)
    #fixhdr_mask = pe.Node(CopyXForm(), name="fixhdr_mask", mem_gb=0.1)

    # Compute masked brain
    #apply_mask = pe.Node(ApplyMask(), name="apply_mask")

    if not pre_mask:
        from nipype.interfaces.ants.utils import AI

        bold_template = get_template(
            "MNI152NLin2009cAsym", resolution=2, desc="fMRIPrep", suffix="boldref"
        )
        brain_mask = get_template(
            "MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask"
        )

        # Initialize transforms with antsAI
        init_aff = pe.Node(
            AI(
                fixed_image=str(bold_template),
                fixed_image_mask=str(brain_mask),
                metric=("Mattes", 32, "Regular", 0.2),
                transform=("Affine", 0.1),
                search_factor=(20, 0.12),
                principal_axes=False,
                convergence=(10, 1e-6, 10),
                verbose=True,
            ),
            name="init_aff",
            n_procs=omp_nthreads,
        )

        # Registration().version may be None
        if parseversion(Registration().version or "0.0.0") > Version("2.2.0"):
            init_aff.inputs.search_grid = (40, (0, 40, 40))

        # Set up spatial normalization
        norm = pe.Node(
            Registration(from_file=data.load("epi_atlasbased_brainmask.json")),
            name="norm",
            n_procs=omp_nthreads,
        )
        norm.inputs.fixed_image = str(bold_template)
        map_brainmask = pe.Node(
            ApplyTransforms(
                interpolation="Linear",
                # Use the higher resolution and probseg for numerical stability in rounding
                input_image=str(
                    get_template(
                        "MNI152NLin2009cAsym",
                        resolution=1,
                        label="brain",
                        suffix="probseg",
                    )
                ),
            ),
            name="map_brainmask",
        )
        # Ensure mask's header matches reference's
        check_hdr = pe.Node(MatchHeader(), name="check_hdr", run_without_submitting=True)

        # fmt: off
        workflow.connect([
            (inputnode, check_hdr, [("in_file", "reference")]),
            (inputnode, init_aff, [("in_file", "moving_image")]),
            (inputnode, map_brainmask, [("in_file", "reference_image")]),
            (inputnode, norm, [("in_file", "moving_image")]),
            (init_aff, norm, [("output_transform", "initial_moving_transform")]),
            (norm, map_brainmask, [
                ("reverse_invert_flags", "invert_transform_flags"),
                ("reverse_transforms", "transforms"),
            ]),
            (map_brainmask, check_hdr, [("output_image", "in_file")]),
            (check_hdr, n4_correct, [("out_file", "weight_image")]),
        ])
        # fmt: on
    else:
        # fmt: off
        workflow.connect([
            (inputnode, n4_correct, [("pre_mask", "weight_image")]),
        ])
        # fmt: on

    # fmt: off
    workflow.connect([
        (inputnode, n4_correct, [("in_file", "input_image")]),
        (n4_correct, synthstrip, [("output_image", "input_file")]),
        (synthstrip, outputnode, [("out_brain", "in_file")]),
        (synthstrip, outputnode, [("out_brain_mask", "in_mask")]),
        (n4_correct, outputnode, [("output_image", "bias_corrected_file")]),
    ])
    # fmt: on

    return workflow