#ifndef ANGLESIFT_HPP
#define  ANGLESIFT_HPP

#ifndef VERBOSE
#define VERBOSE true
#endif

#include <opencv2/opencv.hpp>

class AngleSift
{
	// default number of bins in histogram for vOrientation assignment
	private: const int SIFT_ORI_HIST_BINS = 36;

	// vOrientation vMagnitude relative to max that results in new feature
	private: const double SIFT_ORI_PEAK_RATIO = 0.8;

	private: double orientation = 0.0;
	private: long bins = 8;
	private: long dims = 4;

	public: AngleSift( long dims = 4, long bins = 8, const double orientation = 0.0 )
	{
		this->dims = dims;
		this->bins = bins;
		this->orientation = orientation * M_PI / 180.0;
	}

	public: virtual ~AngleSift(){}

	public: inline void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors ) const
	{
		const double hist_width = 100;
		const double bins_per_rad = bins / M_PI;
		const double exp_denom = std::pow( dims, 2 ) * 0.5;

		descriptors.create( keypoints.size(), std::pow( dims, 2 ) * bins, CV_8UC1 );

		for( unsigned long i = 0; i < keypoints.size(); i++ )
		{
			double omax = calc_feature_oris( image, keypoints[i] );
			double cos_t = std::cos( std::fabs( omax ) );
			double sin_t = std::sin( std::fabs( omax ) );
			double radius = keypoints[i].size / 2.0;

			std::vector<double> vMagnitude( image.channels(), 0.0 );
			std::vector<double> vOrientation( image.channels(), 0.0 );
			std::vector<std::vector<std::vector<double>>> v3Hist( dims, std::vector<std::vector<double>>( dims, std::vector<double>( bins, 0.0 ) ) );

			for( long y = -radius; y <= radius; y++ )
			{
				for( long x = -radius; x <= radius; x++ )
				{
					double colsRotation = ( x * cos_t - y * sin_t ) / hist_width;
					double rowsRotation = ( x * sin_t + y * cos_t ) / hist_width;
					double rowsBin = rowsRotation + dims / 2 - 0.5;
					double colsBin = colsRotation + dims / 2 - 0.5;

					if( rowsBin > -1.0  &&  rowsBin < dims  &&  colsBin > -1.0  &&  colsBin < dims )
					{
						if( calc_grad_mag_ori( image, keypoints[i].pt.x + x, keypoints[i].pt.y + y, vMagnitude, vOrientation ) )
						{
							if( image.channels() == 1 )
							{
								vOrientation[0] -= orientation;

								while( vOrientation[0] < 0.0 ) vOrientation[0] += M_PI;
								while( vOrientation[0] >= M_PI ) vOrientation[0] -= M_PI;

								double orientationBin = vOrientation[0] * bins_per_rad;
								double weight = std::exp( -( std::pow( colsRotation, 2 ) + std::pow( rowsRotation, 2 ) ) / exp_denom );

								interp_hist_entry( v3Hist, colsBin, rowsBin, orientationBin, vMagnitude[0] * weight, dims, bins );
							}
							else if( image.channels() >= 3 )
							{
								for( long c = 0; c < image.channels(); c++ )
								{
									vOrientation[c] -= orientation;

									while( vOrientation[c] < 0.0 ) vOrientation[c] += M_PI;
									while( vOrientation[c] >= M_PI ) vOrientation[c] -= M_PI;

									double orientationBin = vOrientation[c] * bins_per_rad;
									double weight = std::exp( -( std::pow( colsRotation, 2 ) + std::pow( rowsRotation, 2 ) ) / exp_denom );

									interp_hist_entry( v3Hist, colsBin, rowsBin, orientationBin, vMagnitude[c] * weight, dims, bins );
								}
							}
						}
					}
				}
			}

			std::vector<double> descr;
			hist_to_descr( v3Hist, dims, bins, descr );

			for( unsigned long j = 0; j < descr.size(); j++ )
			{
				descriptors.at<uchar>( i, j ) = int( descr[j] );
			}
		}
	}

	/**
	Calculates the gradient vMagnitude and vOrientation at a given pixel.

	@param img image
	@param r pixel row
	@param c pixel col
	@param vMagnitude output as gradient vMagnitude at pixel (r,c)
	@param vOrientation output as gradient vOrientation at pixel (r,c)

	@return Returns 1 if the specified pixel is a valid one and sets vMagnitude and
	vOrientation accordingly; otherwise returns 0
	**/
	private: inline bool calc_grad_mag_ori(
		const cv::Mat image,
		const long x,
		const long y,
		std::vector<double>& vMagnitude,
		std::vector<double>& vOrientation
	) const
	{
		if( y > 0 && y < image.rows - 1 && x > 0 && x < image.cols -1 )
		{
			if( image.channels() == 1 )
			{
				double dx = image.at<uchar>( y, x + 1 ) - image.at<uchar>( y, x - 1 );
				double dy = image.at<uchar>( y + 1, x ) - image.at<uchar>( y - 1, x );

				vMagnitude[0] = std::sqrt( std::pow( dx, 2 ) + std::pow( dy, 2 ) );
				vOrientation[0] = std::atan2( dy, dx );
			}
			else if( image.channels() == 3 )
			{
				for( long c = 0; c < image.channels(); c++ )
				{
					double dx = image.at<cv::Vec3b>( y, x + 1 )[c] - image.at<cv::Vec3b>( y, x - 1 )[c];
					double dy = image.at<cv::Vec3b>( y + 1, x )[c] - image.at<cv::Vec3b>( y - 1, x )[c];

					vMagnitude[c] = std::sqrt( std::pow( dx, 2 ) + std::pow( dy, 2 ) );
					vOrientation[c] = std::atan2( dy, dx );
				}
			}
			else if( image.channels() == 4 )
			{
				for( long c = 0; c < image.channels(); c++ )
				{
					double dx = image.at<cv::Vec4b>( y, x + 1 )[c] - image.at<cv::Vec4b>( y, x - 1 )[c];
					double dy = image.at<cv::Vec4b>( y + 1, x )[c] - image.at<cv::Vec4b>( y - 1, x )[c];

					vMagnitude[c] = std::sqrt( std::pow( dx, 2 ) + std::pow( dy, 2 ) );
					vOrientation[c] = std::atan2( dy, dx );
				}
			}
			else return false;

			return true;
		}
		else return false;
	}

	/**
	Interpolates an entry into the array of vOrientation histograms that form
	the feature descriptor.

	@param hist 2D array of vOrientation histograms
	@param rowsBin sub-bin row coordinate of entry
	@param colsBin sub-bin column coordinate of entry
	@param orientationBin sub-bin vOrientation coordinate of entry
	@param vMagnitude size of entry
	@param d width of 2D array of vOrientation histograms
	@param n number of bins per vOrientation histogram
	**/
	public: inline void interp_hist_entry(
		std::vector<std::vector<std::vector<double>>>& v3Hist,
		const double colsBin,
		const double rowsBin,
		const double orientationBin,
		const double magnitude,
		const int dims,
		const int bins
	) const
	{
		std::vector<std::vector<double>> v2Row;
		std::vector<double> vHist;

		int iRows = (int) rowsBin ;
		int iCols = (int) colsBin ;
		int iOrientation = (int) orientationBin ;
		double dRows = rowsBin - iRows;
		double dCols = colsBin - iCols;
		double dOrientation = orientationBin - iOrientation;

		/*
		The entry is distributed into up to 8 bins.  Each entry into a bin
		is multiplied by a weight of 1 - d for each dimension, where d is the
		distance from the center value of the bin measured in bin units.
		*/
		for( long y = 0; y <= 1; y++ )
		{
			long rb = iRows + y;

			if( rb >= 0  &&  rb < dims )
			{
				double vRows = magnitude * ( ( y == 0 ) ? 1.0 - dRows : dRows );
				v2Row = v3Hist[rb];

				for( long x = 0; x <= 1; x++ )
				{
					long cb = iCols + x;

					if( cb >= 0  &&  cb < dims )
					{
						double vCols = vRows * ( ( x == 0 ) ? 1.0 - dCols : dCols );
						vHist = v2Row[cb];

						for( long orientation = 0; orientation <= 1; orientation++ )
						{
							long ob = ( iOrientation + orientation ) % bins;
							double vOrientation = vCols * ( ( orientation == 0 ) ? 1.0 - dOrientation : dOrientation );

							vHist[ob] += vOrientation;
						}

						v2Row[cb] = vHist;
					}
				}

				v3Hist[rb] = v2Row;
			}
		}
	}

	/*
	Computes a canonical vOrientation for each image feature in an array.  Based
	on Section 5 of Lowe's paper.  This function adds features to the array when
	there is more than one dominant vOrientation at a given feature location.

	@param features an array of image features
	@param gauss_pyr Gaussian scale space pyramid
	*/
	public: inline double calc_feature_oris(
		const cv::Mat image,
		const cv::KeyPoint keypoint
	) const
	{
		std::vector<double> hist = ori_hist( image, keypoint, SIFT_ORI_HIST_BINS, 0.0 );

		for( int i = 0; i < 2; i++ ) smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );

		double omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );
		omax = add_good_ori_features( hist, SIFT_ORI_HIST_BINS, omax * SIFT_ORI_PEAK_RATIO );

		return omax;
	}

	/*
	Computes a gradient vOrientation histogram at a specified pixel.

	@param img image
	@param r pixel row
	@param c pixel col
	@param n number of histogram bins
	@param rad radius of region over which histogram is computed
	@param sigma std for Gaussian weighting of histogram entries

	@return Returns an n-element array containing an vOrientation histogram
	representing orientations between 0 and 2 PI.
	*/
	private: inline std::vector<double> ori_hist(
		const cv::Mat image,
		const cv::KeyPoint keypoint,
		const long bins,
		const double sigma
	) const
	{
		std::vector<double> hist( bins );
		std::vector<double> vMagnitude( image.channels(), 0.0 );
		std::vector<double> vOrientation( image.channels(), 0.0 );

		long radius = keypoint.size / 2.0;

		double PI2 = M_PI * 2.0;
		double exp_denom = 0.5;

		for( long y = -radius; y <= radius; y++ )
		{
			for( long x = -radius; x <= radius; x++ )
			{
				if( calc_grad_mag_ori( image, keypoint.pt.x + x, keypoint.pt.y + y, vMagnitude, vOrientation ) )
				{
					double weight = std::exp( -( std::pow( x, 2 ) + std::pow( y, 2 ) ) / exp_denom );

					for( long c = 0; c < image.channels(); c++ )
					{
						double bin = ( bins * ( vOrientation[c] + M_PI ) / PI2 );
						bin = std::ceil( ( bin < bins ) ? bin : 0 + 0.5 );
						hist[int( bin )] += ( weight * vMagnitude[c] );
					}
				}
			}
		}

		return hist;
	}

	/*
	Gaussian smooths an vOrientation histogram.

	@param hist an vOrientation histogram
	@param n number of bins
	*/
	private: inline void smooth_ori_hist(
		std::vector<double>& hist,
		const long bins
	) const
	{
		double tmp;
		double h0 = hist[0];
		double prev = hist[bins - 1];

		for( long i = 0; i < bins; i++ )
		{
			tmp = hist[i];
			hist[i] = ( 0.25 * prev ) + ( 0.5 * hist[i] ) + ( 0.25 * ( i + 1 == bins ) ? h0 : hist[i + 1] );
			prev = tmp;
		}
	}

	/*
	Finds the vMagnitude of the dominant vOrientation in a histogram

	@param hist an vOrientation histogram
	@param n number of bins

	@return Returns the value of the largest bin in hist
	*/
	private: inline double dominant_ori(
		const std::vector<double> hist,
		const long bins
	) const
	{
		double omax = hist[0];

		for( long i = 1; i < bins; i++ )
		{
			omax = std::max( omax, hist[i] );
		}

		return omax;
	}

	/*
	Adds features to an array for every vOrientation in a histogram greater than
	a specified threshold.

	@param features new features are added to the end of this array
	@param hist vOrientation histogram
	@param n number of bins in hist
	@param mag_thr new features are added for entries in hist greater than this
	@param feat new features are clones of this with different orientations
	*/
	private: inline double add_good_ori_features(
		const std::vector<double> hist,
		const long bins,
		const double magnitude_threshold
	) const
	{
		double orientation = 0;
		double PI2 = M_PI * 2.0;

		for( long i = 0; i < bins; i++ )
		{
			int prev = ( i == 0 ) ? bins - 1 : i - 1;
			int next = ( i + 1 ) % bins;

			if( hist[i] > hist[prev]  &&  hist[i] > hist[next]  &&  hist[i] >= magnitude_threshold )
			{
				double bin = i + interp_hist_peak( hist[prev], hist[i], hist[next] );
				bin = ( bin < 0 ) ? bins + bin : ( bin >= bins ) ? bin - bins : bin;
				orientation = ( ( PI2 * bin ) / bins ) - M_PI;
			}
		}

		return orientation;
	}

	private: inline double interp_hist_peak(
		const double prev,
		const double current,
		const double next
	) const
	{
		return ( 0.5 * ( ( prev ) - ( next ) ) / ( ( prev ) - 2.0 * ( current ) + ( next ) ) );
	}

	/**
	Converts the 2D array of vOrientation histograms into a feature's descriptor
	vector.

	@param hist 2D array of vOrientation histograms
	@param d width of hist
	@param n bins per histogram
	@param feat feature into which to store descriptor
	**/
	private: inline void hist_to_descr(
		const std::vector<std::vector<std::vector<double>>> v3Hist,
		const int dims,
		const int bins,
		std::vector<double>& descriptor
	) const
	{
		for( long y = 0; y < dims; y++ )
		{
			for( long x = 0; x < dims; x++ )
			{
				for( long o = 0; o < bins; o++ )
				{
					descriptor.push_back( v3Hist[y][x][o] );
				}
			}
		}

		normalize_descr( descriptor );

		// threshold on vMagnitude of elements of descriptor vector
		//#define SIFT_DESCR_MAG_THR 0.2

		for( unsigned long i = 0; i < descriptor.size(); i++ )
		{
			if( descriptor[i] > 0.5d ) descriptor[i] = 0.5d;
		}

		normalize_descr( descriptor );

		//  convert floating-point descriptor to integer valued descriptor
		//  factor used to convert floating-point descriptor to unsigned char
		//#define SIFT_INT_DESCR_FCTR 512.0

		for( unsigned long i = 0; i < descriptor.size(); i++ )
		{
			double int_val = std::ceil( ( 512 * descriptor[i] ) + 0.5 );
			descriptor[i] = std::min( 255.0, int_val );
		}
	}

	/**
	Normalizes a feature's descriptor vector to unitl length

	@param feat feature
	**/
	public: inline void normalize_descr( std::vector<double>& descriptor ) const
	{
		double len_sq = 0.0;

		for( unsigned long i = 0; i < descriptor.size(); i++ )
		{
			double current = descriptor[i];
			len_sq += std::pow( current, 2 );
		}

		double len_inv = 1.0 / std::sqrt( len_sq );

		for( unsigned long i = 0; i < descriptor.size(); i++ )
		{
			descriptor[i] *= len_inv;
		}
	}
};

#endif // ANGLESIFT_HPP
