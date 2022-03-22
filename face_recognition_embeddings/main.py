from lib.Pipeline import DataSet
import argparse	

# Slope .5
# Intercept .178
# Known Acc 88.45%
# Unknown Acc 81.62%

# Slope .5
# Intercept .19
# Known Acc 90.42%
# Unknown Acc 76.01%

# Slope .5
# Intercept .164
# Known Acc 85.15%
# Unknown Acc 85.66%

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",			"-dir",	help="Dataset directory",			type=str,	default="datasets/LookDataSet")
	parser.add_argument("--extension",			"-ext",	help="Dataset images extension",	type=str,	default="jpg")
	parser.add_argument("--size",				"-si",	help="Image size",					type=int,	default=40)
	parser.add_argument("--slope_limit",		"-sl",	help="Slope Limit",					type=float, default=.5)
	parser.add_argument("--intercept_limit",	"-il",	help="Intercept Limit",				type=float,	default=.164)
	args = parser.parse_args()
	
	ds = DataSet(
		directory=args.directory,
        extension=args.extension,
        size=args.size,
		slope_limit=args.slope_limit,
		intercept_limit=args.intercept_limit
    )

	ds.print_dataset_info()
	ds.load_model(name='model_v1',train=True)
	
	#filename='datasets/LookDataSet/Test/Emma_Watson/Emma_Watson_018_resized.jpg'
	#ds.single_image(filename=filename)
	
	#ds.test_model(graphs=False,print_detail=False)
	
	ds.testing_webcam()


if __name__ == "__main__":
    main()
