# Docker command for the old docker lengh2/Omni_seg:
# docker run -shm-size 64G -it -rm -v $input_dir:/INPUTS/ -v $output_dir:/OUTPUTS lengh2/omni_seg

# The old docker (lengh2/omni_seg) also runs this:
# cd /Omni-Seg/Omni_seg_pipeline_gpu/apex
# python3 run setup.py install

# This will be done outside the docker - input is structured by the script
# python '/content/drive/My Drive/Omni-Seg/Omni_seg_pipeline_gpu/svs_input/svs_to_png.py'

# All these scripts refer to input folder
cd '/Omni-Seg/Omni_seg_pipeline_gpu'

python '1024_Step1_GridPatch_overlap_padding.py'

python '1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py'

python Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path '/Omni-Seg/Omni_seg_pipeline_gpu/snapshots_2D/fold1_with_white_Omni-Seg_normalwhole_1201/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth'

python 'step3.py'

python 'step4.py'

cp -r final_merge /OUTPUTS