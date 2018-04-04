#ifndef SIFTDESCRIPTOR_HPP
#define  SIFTDESCRIPTOR_HPP

#include <opencv2/opencv.hpp>

class SiftDescriptor
{
	public: double sift_ori=0.0d;
	public: double THR = 10.0d;
	/* default number of bins in histogram for orientation assignment */
	public: int SIFT_ORI_HIST_BINS = 36;
	/* orientation magnitude relative to max that results in new feature */
	public: double SIFT_ORI_PEAK_RATIO = 0.8;

	public: inline void setOri( double ori )
	{
		sift_ori = ori;
	}

	public: inline double getOri()
	{
		return sift_ori;
	}

	/*
	Calculates the gradient magnitude and orientation at a given pixel.

	@param img image
	@param r pixel row
	@param c pixel col
	@param mag output as gradient magnitude at pixel (r,c)
	@param ori output as gradient orientation at pixel (r,c)

	@return Returns 1 if the specified pixel is a valid one and sets mag and
	ori accordingly; otherwise returns 0
	*/

	public: inline bool calc_grad_mag_ori( cv::Mat img, long rows, long cols, std::vector<double> mag, std::vector<double> ori )
	{
		double dx,dy;
		int height=img.cols;
		int width=img.rows;

		if( rows > 0 && rows < height - 1 && cols > 0 && cols < width -1 )
		{
			dx = img.at<uchar>( rows, cols + 1 ) - img.at<uchar>( rows, cols - 1 );
			dy = img.at<uchar>( rows - 1, cols ) - img.at<uchar>( rows + 1, cols );

			mag[0] = std::sqrt( dx * dx + dy * dy );

			//ori[0] = myMathFunctionAtan(dy,dx);
			//ori[0] = myMathFunctionAtan180(dy,dx);
			//ori[0] = myMathFunctionAtan360(dy,dx);

			ori[0] = std::atan2(dy,dx);
			//ori[0] = std::atan(dy/dx);

			//std::cout << "@["+r+"]["+c+"]="+mag[0]+"\t"+ori[0]);
			return true;
		}else
		return false;
	}

	/*
	Interpolates an entry into the array of orientation histograms that form
	the feature descriptor.

	@param hist 2D array of orientation histograms
	@param rbin sub-bin row coordinate of entry
	@param cbin sub-bin column coordinate of entry
	@param obin sub-bin orientation coordinate of entry
	@param mag size of entry
	@param d width of 2D array of orientation histograms
	@param n number of bins per orientation histogram
	*/
	public: inline void interp_hist_entry( std::vector<std::vector<std::vector<double>>> hist, double rbin, double cbin, double obin, double mag, int d, int n )
	{
		double d_r, d_c, d_o, v_r, v_c, v_o;
		std::vector<std::vector<double>> row;
		std::vector<double> h;
		int r0, c0, o0, rb, cb, ob, r, c, o;

		r0 = (int) rbin ;
		c0 = (int) cbin ;
		o0 = (int) obin ;
		d_r = rbin - r0;
		d_c = cbin - c0;
		d_o = obin - o0;

		/*
		The entry is distributed into up to 8 bins.  Each entry into a bin
		is multiplied by a weight of 1 - d for each dimension, where d is the
		distance from the center value of the bin measured in bin units.
		*/
		for( r = 0; r <= 1; r++ )
		{
			rb = r0 + r;
			if( rb >= 0  &&  rb < d )
			{
				v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
				row = hist[rb];
				for( c = 0; c <= 1; c++ )
				{
					cb = c0 + c;
					if( cb >= 0  &&  cb < d )
					{
						v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
						h = row[cb];
						for( o = 0; o <= 1; o++ )
						{
							ob = ( o0 + o ) % n;
							v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
							h[ob] += v_o;
						}
					}
				}
			}
		}
	}

	/*
	Computes the 2D array of orientation histograms that form the feature
	descriptor.  Based on Section 6.1 of Lowe's paper.

	@param img image used in descriptor computation
	@param r row coord of center of orientation histogram array
	@param c column coord of center of orientation histogram array
	@param ori canonical orientation of feature whose descr is being computed
	@param scl scale relative to img of feature whose descr is being computed
	@param d width of 2d array of orientation histograms
	@param n bins per orientation histogram

	@return Returns a d x d array of n-bin orientation histograms.
	*/

	public: inline std::vector<std::vector<std::vector<double>>> descr_hist180( cv::Mat img, long rows, long cols, int d, int n )
	{
		std::vector<std::vector<std::vector<double>>> hist;
		double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot,
		w, rbin, cbin, obin, bins_per_rad, PI = M_PI,ori=0.0;

		int radius;
		std::vector<double> grad_ori = {0.0};
		std::vector<double> grad_mag = {0.0};

		hist.resize( d );

		for( int i = 0; i < d; i++ )
		{
			hist[i].resize( d );
			for( int j=0; j<d; j++)
			hist[i][j].resize( n );
		}

		double omax = calc_feature_oris( img, rows, cols );

		cos_t = std::cos( std::fabs(omax) );
		sin_t = std::sin( std::fabs(omax) );

		bins_per_rad = n / PI;
		exp_denom = d * d * 0.5;

		/* determines the size of a single descriptor orientation histogram */
		//#define SIFT_DESCR_SCL_FCTR 3.0
		hist_width = 100;// 360/8;3.0d * 1;

		//radius = (int)(hist_width * std::sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5);
		radius = std::min( img.cols / 2, img.rows / 2 );
		std::cout << "radius : " << radius << std::endl;

		for(int i = -radius; i <= radius; i++)
		for(int j = -radius; j <= radius; j++)
		{
			/*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
			*/
			c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			rbin = r_rot + d / 2 - 0.5;
			cbin = c_rot + d / 2 - 0.5;

			if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
			{
				if( calc_grad_mag_ori( img, rows + i, cols + j, grad_mag, grad_ori ) )
				{
					grad_ori[0] -= ori;
					while( grad_ori[0] < 0.0 )
					grad_ori[0] += PI;

					while( grad_ori[0] >= PI )
					grad_ori[0] -= PI;

					obin = grad_ori[0] * bins_per_rad;
					w = std::exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
					interp_hist_entry( hist, rbin, cbin, obin, grad_mag[0] * w, d, n );
				}
			}
		}

		return hist;
	}

	public: inline std::vector<std::vector<std::vector<double>>> descr_hist360( cv::Mat img, long rows, long cols, int d, int n )
	{
		std::vector<std::vector<std::vector<double>>> hist;
		double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot,
		w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * M_PI,ori=0.0;

		int radius;
		std::vector<double> grad_ori={0.0};
		std::vector<double> grad_mag={0.0};

		hist.resize( d );

		for( int i = 0; i < d; i++ )
		{
			hist[i].resize( d );
			for( int j=0; j<d; j++)
			hist[i][j].resize( n );
		}

		double omax = calc_feature_oris( img, rows, cols );

		cos_t = std::cos( std::fabs(omax) );
		sin_t = std::sin( std::fabs(omax) );

		bins_per_rad = n / PI2;
		exp_denom = d * d * 0.5;

		/* determines the size of a single descriptor orientation histogram */
		//#define SIFT_DESCR_SCL_FCTR 3.0
		hist_width = 100;// 360/8;3.0d * 1;

		//radius = (int)(hist_width * std::sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5);
		radius = (int)std::min(img.cols/2,img.rows/2);
		std::cout << "radius : " << radius << std::endl;

		for(int i = -radius; i <= radius; i++)
		for(int j = -radius; j <= radius; j++)
		{
			/*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
			*/
			c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			rbin = r_rot + d / 2 - 0.5;
			cbin = c_rot + d / 2 - 0.5;

			if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
			if( calc_grad_mag_ori( img, rows + i, cols + j, grad_mag, grad_ori ) )
			{

				//if( grad_mag[0] < THR ) continue;

				grad_ori[0] -= ori;
				while( grad_ori[0] < 0.0 )
				grad_ori[0] += PI2;

				while( grad_ori[0] >= PI2 )
				grad_ori[0] -= PI2;


				obin = grad_ori[0] * bins_per_rad;
				w = std::exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
				interp_hist_entry( hist, rbin, cbin, obin, grad_mag[0] * w, d, n );
			}
		}

		return hist;
	}

	/*
	Converts the 2D array of orientation histograms into a feature's descriptor
	vector.

	@param hist 2D array of orientation histograms
	@param d width of hist
	@param n bins per histogram
	@param feat feature into which to store descriptor
	*/
	public: inline void hist_to_descr( std::vector<std::vector<std::vector<double>>> hist, int d, int n, std::vector<double>& descr )
	{
		int int_val, i, r, c, o, k = 0;

		for( r = 0; r < d; r++ )
		for( c = 0; c < d; c++ )
		for( o = 0; o < n; o++ )
		descr[k++] = hist[r][c][o];

		normalize_descr( descr,k );

		/* threshold on magnitude of elements of descriptor vector */
		//#define SIFT_DESCR_MAG_THR 0.2

		for( i = 0; i < k; i++ )
		if( descr[i] > 0.5d )
		descr[i] = 0.5d;


		normalize_descr( descr, k );

		/* convert floating-point descriptor to integer valued descriptor */
		/* factor used to convert floating-point descriptor to unsigned char */
		//#define SIFT_INT_DESCR_FCTR 512.0

		for( i = 0; i < k; i++ )
		{
			int_val = (int)(512 * descr[i]);
			descr[i] = std::min( 255, int_val );
		}
	}


	/*
	Normalizes a feature's descriptor vector to unitl length

	@param feat feature
	*/
	public: inline void normalize_descr( std::vector<double>& descr, int& len )
	{
		double cur, len_inv, len_sq = 0.0;

		for( int i = 0; i < len; i++ )
		{
			cur = descr[i];
			len_sq += cur*cur;
		}

		len_inv = 1.0 / std::sqrt( len_sq );
		for(int i = 0; i < len; i++ )
		descr[i] *= len_inv;
	}

	public: inline std::vector<double> getFeatureVector( cv::Mat image )
	{
		int rows = image.cols / 2;
		int cols = image.rows / 2;

		std::vector<std::vector<std::vector<double>>> hist = descr_hist180( image, rows, cols, 4, 8 );
		std::vector<double> feat( 128 );

		hist_to_descr( hist, 4, 8, feat );

		return feat;
	}

	public: inline std::vector<double> getFeatureVector( cv::Mat dis, int angle )
	{
		int rows = dis.rows / 2;
		int cols = dis.cols / 2;

		cv::Mat img;
		cvtColor( img, dis, cv::COLOR_BGR2GRAY );

		std::vector<std::vector<std::vector<double>>> hist;
		if( angle == 180 )
		hist = descr_hist180( img, rows, cols, 4, 8 );
		else
		hist = descr_hist360( img, rows, cols, 4, 8 );

		std::vector<double> feat( 128 );

		hist_to_descr( hist, 4, 8, feat );

		return feat;

	}

	public: inline int myMathFunctionAtan(double p, double q){
		double pi=M_PI;
		int degree;
		double theta=std::atan(q/p);

		if(p<0)
		theta+=pi;
		if(theta<0)
		theta+=2*pi;
		if(theta>2*pi)
		theta-=2*pi;
		degree=(int)(360*theta/(2*pi));

		return degree;
	}

	public: inline int myMathFunctionAtan180(double p, double q){
		double pi=M_PI;
		int degree;
		double theta=std::atan(q/p);

		degree = (int)(((theta + (pi/2))*180)/pi);

		return degree;
	}

	public: inline int myMathFunctionAtan360(double p, double q){
		double pi=M_PI;
		int degree;
		double theta=std::atan2(q,p);

		degree = (int)(((theta + pi)*180)/pi);

		return degree;
	}


	/*
	Computes a canonical orientation for each image feature in an array.  Based
	on Section 5 of Lowe's paper.  This function adds features to the array when
	there is more than one dominant orientation at a given feature location.

	@param features an array of image features
	@param gauss_pyr Gaussian scale space pyramid
	*/
	public: inline double calc_feature_oris( cv::Mat img, long rows, long cols )
	{
		double omax;
		int radius = std::min( img.cols / 2, img.rows / 2 );
		std::vector<double> hist = ori_hist( img, rows, cols, SIFT_ORI_HIST_BINS, radius, 0.0 );

		/* number of passes of orientation histogram smoothing */
		//#define SIFT_ORI_SMOOTH_PASSES 2
		/* default number of bins in histogram for orientation assignment */
		//#define SIFT_ORI_HIST_BINS 36

		for( int j = 0; j < 2; j++ )
		smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );
		omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );
		omax = add_good_ori_features( hist, SIFT_ORI_HIST_BINS,omax*SIFT_ORI_PEAK_RATIO);

		return omax;
	}

	/*
	Computes a gradient orientation histogram at a specified pixel.

	@param img image
	@param r pixel row
	@param c pixel col
	@param n number of histogram bins
	@param rad radius of region over which histogram is computed
	@param sigma std for Gaussian weighting of histogram entries

	@return Returns an n-element array containing an orientation histogram
	representing orientations between 0 and 2 PI.
	*/
	private: inline std::vector<double> ori_hist( cv::Mat img, long rows, long cols, int n, int rad, double sigma )
	{
		std::vector<double> hist;
		double w, exp_denom, PI2 = M_PI * 2.0;
		int bin, i, j;

		std::vector<double> mag = {0.0d};
		std::vector<double> ori = {0.0d};
		hist.resize( n );
		exp_denom = 0.5;//2.0 * sigma * sigma;
		for( i = -rad; i <= rad; i++ )
		for( j = -rad; j <= rad; j++ )

		if( calc_grad_mag_ori( img, rows + i, cols + j, mag, ori ) )
		{
			w = std::exp( -( i*i + j*j ) / exp_denom );
			bin = (int)( n * ( ori[0] + M_PI ) / PI2 );
			bin = ( bin < n )? bin : 0;
			hist[bin] += w * mag[0];
		}

		return hist;
	}

	/*
	Gaussian smooths an orientation histogram.

	@param hist an orientation histogram
	@param n number of bins
	*/
	private: inline void smooth_ori_hist( std::vector<double> hist, int n )
	{
		double prev, tmp, h0 = hist[0];
		int i;

		prev = hist[n-1];
		for( i = 0; i < n; i++ )
		{
			tmp = hist[i];
			hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
			prev = tmp;
		}
	}

	/*
	Finds the magnitude of the dominant orientation in a histogram

	@param hist an orientation histogram
	@param n number of bins

	@return Returns the value of the largest bin in hist
	*/
	private: inline double dominant_ori( std::vector<double> hist, int n )
	{
		double omax;
		int i;

		omax = hist[0];
		for( i = 1; i < n; i++ )
		if( hist[i] > omax )
		{
			omax = hist[i];
		}

		return omax;
	}

	/*
	Adds features to an array for every orientation in a histogram greater than
	a specified threshold.

	@param features new features are added to the end of this array
	@param hist orientation histogram
	@param n number of bins in hist
	@param mag_thr new features are added for entries in hist greater than this
	@param feat new features are clones of this with different orientations
	*/
	private: inline double add_good_ori_features( std::vector<double> hist, int n,double mag_thr )
	{
		double ori=0,bin, PI2 = M_PI * 2.0;
		int l, r, i;

		for( i = 0; i < n; i++ )
		{
			l = ( i == 0 )? n - 1 : i-1;
			r = ( i + 1 ) % n;

			if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
			{
				bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
				bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
				ori = ( ( PI2 * bin ) / n ) - M_PI;
			}
		}
		return ori;
	}

	private: inline double interp_hist_peak( double l, double c, double r ) {
		return ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) );
	}

	public: inline void Test()
	{

		cv::Mat input = cv::imread( "/home/dekwan/Pictures/Sarah/IMG-20170821-WA0112.jpg" );

		std::vector<std::vector<std::vector<double>>> hist180 = descr_hist180( input, input.rows / 2, input.cols / 2, 4, 8 );
		std::vector<std::vector<std::vector<double>>> hist360 = descr_hist360( input, input.rows / 2, input.cols / 2, 4, 8 );
	}
};

#endif // SIFTDESCRIPTOR_HPP
