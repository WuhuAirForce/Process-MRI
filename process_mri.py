from monai.transforms import *
import SimpleITK as sitk
import argparse


def main(args):
    # load
    case_list = [args.input]
    raw_data = sitk.ReadImage(case_list[0])
    print(raw_data)
    # process
    pixel_min,pixel_max=-1000,400
    space = (1.5,1.5,1.5)
    img_size = (96,96,96)
    process_dict = {'detection':CropForeground(),
                    'blur':RandGaussianNoise(),
                    'threshold':ScaleIntensityRange(a_min=pixel_min, a_max=pixel_max, b_min=0, b_max=1, clip=True),
                    'rescaling':Resize( spatial_size=img_size),}
    
    process_list = [LoadImage(),
                    EnsureChannelFirst(),
                    Orientation(axcodes="RAS"),]
    
    for p in args.process:
        process_list.append(process_dict[p])
    print(process_list)
    process = Compose(process_list)
    process_data = process(case_list[0])

    # save
    saver = SaveImage(output_dir=args.output,)
    saver(process_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./sub-0002_ses-01_T1w.nii.gz', help="the port number.",)
    parser.add_argument("--output", type=str, default = './processed' ,)
    parser.add_argument("--process",  nargs='+', default = ['detection','blur','threshold','rescaling'], required=False, )
    parser.add_argument("--process_param",  default=False,required=False, )
    args = parser.parse_args()
    main(args) 
    # raw_data = sitk.ReadImage('./processed/sub-0002_ses-01_T1w/sub-0002_ses-01_T1w_trans.nii.gz')
    # print(raw_data)