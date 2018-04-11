#pragma once

#ifndef  MPEG7_HPP
#define  MPEG7_HPP

#include <memory>
#include "Feature.h"

namespace dekwan
{
	class MPEG7
	{
		protected: std::shared_ptr<Frame> frame;
		protected: cv::Mat image;

		// MPEG7
		protected: bool imgFlag = true;
		protected: bool grayFlag = true;
		protected: bool maskFlag = true;

		public: MPEG7()
		{
			this->frame.reset( new Frame( 256, 256, this->imgFlag, this->grayFlag, this->maskFlag ) );
		}

		public: virtual ~MPEG7(){}

		protected: cv::Mat CropKeypoints( const cv::Mat image, const cv::KeyPoint keypoint ) const
		{
			double radius = keypoint.size / 2.0;
			double x = keypoint.pt.x - radius < 0 ? 0 : keypoint.pt.x - radius;
			double y = keypoint.pt.y - radius < 0 ? 0 : keypoint.pt.y - radius;
			double width = x + keypoint.size < image.cols ? keypoint.size : image.cols - x;
			double height = y + keypoint.size < image.rows ? keypoint.size : image.rows - y;

			return image( cv::Rect( x, y, width, height  ) );
		}
	};

	class ColorStructureDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ColorStructureDescriptor> desc;
		private: int descSize = 64;

		public: ColorStructureDescriptor( const int descSize = 64 )
		{
			desc.reset( new XM::ColorStructureDescriptor() );
			this->descSize = descSize;
		}

		public: virtual ~ColorStructureDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), this->descSize, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame->setImage( this->image );
				this->desc = Feature::getColorStructureD( this->frame, this->descSize );

				for( unsigned long x = 0; x < this->desc->GetSize(); x++ )
				{
					descriptors.at<uchar>( y, x ) = uchar( this->desc->GetElement( x ) );
				}
			}
		}
	};

	class ScalableColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ScalableColorDescriptor> desc;
		private: int numCoeff = 256;
		private: int bitPlanesDiscarded = 0;

		public: ScalableColorDescriptor( const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			desc.reset( new XM::ScalableColorDescriptor() );
			this->numCoeff = numCoeff;
			this->bitPlanesDiscarded = bitPlanesDiscarded;
		}

		public: virtual ~ScalableColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), this->numCoeff, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame->setImage( this->image );
				this->desc = Feature::getScalableColorD( this->frame, this->maskFlag, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->desc->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<char>( y, x ) = char( this->desc->GetCoefficient( x ) );
				}
			}
		}
	};

	class GoFGoPColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ScalableColorDescriptor> desc;
		private: int numCoeff = 256;
		private: int bitPlanesDiscarded = 0;

		public: GoFGoPColorDescriptor( const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			desc.reset( new XM::ScalableColorDescriptor() );
			this->numCoeff = numCoeff;
			this->bitPlanesDiscarded = bitPlanesDiscarded;
		}

		public: virtual ~GoFGoPColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), this->numCoeff, CV_8UC1 );
			std::vector<cv::Mat> vImage( 2 );
			vImage[0] = image.clone();

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				vImage[1] = CropKeypoints( image, keypoints[y] );
				this->desc = Feature::getGoFColorD( vImage, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->desc->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<char>( y, x ) = char( this->desc->GetCoefficient( x ) );
				}
			}
		}
	};

	class DominantColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::DominantColorDescriptor> desc;
		private: bool normalize = true;
		private: bool variance = true;
		private: bool spatial = true;
		private: int bin1 = 32;
		private: int bin2 = 32;
		private: int bin3 = 32;

		public: DominantColorDescriptor( const bool normalize = true, const bool variance = true, const bool spatial = true, const int bin1 = 32, const int bin2 = 32, const int bin3 = 32 )
		{
			desc.reset( new XM::DominantColorDescriptor() );
			this->normalize = normalize;
			this->variance = variance;
			this->spatial = spatial;
			this->bin1 = bin1;
			this->bin2 = bin2;
			this->bin3 = bin3;
		}

		public: virtual ~DominantColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{

			descriptors = cv::Mat::zeros( keypoints.size(), ( 8 * 7 ) + 1, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame->setImage( this->image );
				this->desc = Feature::getDominantColorD( this->frame, this->normalize, this->variance, this->spatial, this->bin1, this->bin2, this->bin3 );

				long x = 0;
				descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetSpatialCoherency() );

				for( long i = 0; i < this->desc->GetDominantColorsNumber(); i++ )
				{
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_Percentage );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorValue[0] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorValue[1] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorValue[2] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorVariance[0] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorVariance[1] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColors()[i].m_ColorVariance[2] );
				}
			}
		}
	};

	class ColorLayoutDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ColorLayoutDescriptor> desc;
		private: int numberOfYCoeff = 64;
		private: int numberOfCCoeff = 28;

		public: ColorLayoutDescriptor( const int numberOfYCoeff = 64, const int numberOfCCoeff = 28 )
		{
			desc.reset( new XM::ColorLayoutDescriptor() );
			this->numberOfYCoeff = numberOfYCoeff;
			this->numberOfCCoeff = numberOfCCoeff;
		}

		public: virtual ~ColorLayoutDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{

			descriptors = cv::Mat::zeros( keypoints.size(), this->numberOfYCoeff + ( this->numberOfCCoeff * 2 ), CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame->setImage( this->image );
				this->desc = Feature::getColorLayoutD( this->frame, this->numberOfYCoeff, this->numberOfCCoeff );

				long x = 0;

				for( long i = 0; i < this->desc->GetNumberOfYCoeff(); i++ )
				{
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetYCoeff()[i] );
				}

				for( long i = 0; i < this->desc->GetNumberOfCCoeff(); i++ )
				{
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetCbCoeff()[i] );
					descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetCrCoeff()[i] );
				}
			}
		}
	};

	class EdgeHistogramDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::EdgeHistogramDescriptor> desc;

		public: EdgeHistogramDescriptor()
		{
			desc.reset( new XM::EdgeHistogramDescriptor() );
		}

		public: virtual ~EdgeHistogramDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), 80, CV_32FC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame->setImage( this->image );
				this->desc = Feature::getEdgeHistogramD( this->frame );

				for( unsigned long x = 0; x < 80; x++ )
				{
					descriptors.at<float>( y, x ) = float( this->desc->GetEdgeHistogramD()[x] );
				}
			}
		}
	};

	class HomogeneousTextureDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::HomogeneousTextureDescriptor> desc;
		private: bool layerFlag = true;

		public: HomogeneousTextureDescriptor( const bool layerFlag = true )
		{
			desc.reset( new XM::HomogeneousTextureDescriptor() );
			this->layerFlag = layerFlag;
		}

		public: virtual ~HomogeneousTextureDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{

			descriptors = cv::Mat::zeros( keypoints.size(), 62, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );

				if( this->image.cols < 128 or this->image.rows < 128 )
				{
					cv::resize( this->image, this->image, cv::Size( 128, 128 ) );
				}

				if( this->image.channels() != 1 )
				{
					cvtColor( this->image, this->image, CV_BGR2GRAY );
				}

				this->frame->setGray( this->image );
				this->desc = Feature::getHomogeneousTextureD( this->frame, this->layerFlag );

				for( unsigned long x = 0; x < 62; x++ )
				{
					descriptors.at<uchar>( y, x ) = uchar( this->desc->GetHomogeneousTextureFeature()[x] );
				}
			}
		}
	};

	class ContourShapeDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ContourShapeDescriptor> desc;

		private: double ratio = 3.0;
		private: double threshold1 = 50;
		private: double threshold2;
		private: int apertureSize = 3;
		private: int kernel = 3;

		public: ContourShapeDescriptor( const double ratio = 3.0, const double threshold1 = 50, const int apertureSize = 3, const int kernel = 3 )
		{
			desc.reset( new XM::ContourShapeDescriptor() );

			this->ratio = ratio;
			this->threshold1 = threshold1;
			this->threshold2 = this->threshold1 * this->ratio;
			this->apertureSize = apertureSize;
			this->kernel = kernel;
		}

		public: virtual ~ContourShapeDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), 2 + 2 + ( CONTOURSHAPE_CSSPEAKMASK * 2 ), CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );

				if( this->image.channels() != 1 )
				{
					/// Convert the image to grayscale
  					cvtColor( this->image, this->image, CV_BGR2GRAY );
				}

				/// Reduce noise with a kernel 3x3
				cv::blur( this->image, this->image, cv::Size( this->kernel, this->kernel ) );

				/// Canny detector
				Canny( this->image, this->image, this->threshold1, this->threshold2, this->apertureSize );

				this->frame->setMask( this->image, 255 );
				this->desc = Feature::getContourShapeD( this->frame );

				unsigned long lgcv[2];
				this->desc->GetGlobalCurvature( lgcv[0], lgcv[1] );

				long x = 0;
				descriptors.at<uchar>( y, x++ ) = uchar( lgcv[0] );
				descriptors.at<uchar>( y, x++ ) = uchar( lgcv[1] );

				long noOfPeak = this->desc->GetNoOfPeaks();
				unsigned long lpcv[2];

				this->desc->GetPrototypeCurvature( lpcv[0], lpcv[1] );

				descriptors.at<uchar>( y, x++ ) = uchar( lpcv[0] );
				descriptors.at<uchar>( y, x++ ) = uchar( lpcv[1] );

				for( long i = 0; i < noOfPeak; i++ )
				{
					unsigned short xp, yp;
					this->desc->GetPeak( i, xp, yp );

					descriptors.at<uchar>( y, x++ ) = uchar( xp );
					descriptors.at<uchar>( y, x++ ) = uchar( yp );
				}
			}
		}
	};

	class RegionShapeDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::RegionShapeDescriptor> desc;

		private: double ratio = 3.0;
		private: double threshold1 = 50;
		private: double threshold2;
		private: int apertureSize = 3;
		private: int kernel = 3;

		public: RegionShapeDescriptor( const double ratio = 3.0, const double threshold1 = 50, const int apertureSize = 3, const int kernel = 3 )
		{
			desc.reset( new XM::RegionShapeDescriptor() );

			this->ratio = ratio;
			this->threshold1 = threshold1;
			this->threshold2 = this->threshold1 * this->ratio;
			this->apertureSize = apertureSize;
			this->kernel = kernel;
		}

		public: virtual ~RegionShapeDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), ART_ANGULAR * ART_RADIAL, CV_32FC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );

				if( this->image.channels() != 1 )
				{
					/// Convert the image to grayscale
  					cvtColor( this->image, this->image, CV_BGR2GRAY );
				}

				/// Reduce noise with a kernel 3x3
				cv::blur( this->image, this->image, cv::Size( this->kernel, this->kernel ) );

				/// Canny detector
				Canny( this->image, this->image, this->threshold1, this->threshold2, this->apertureSize );

				this->frame->setMask( this->image, 255 );
				this->desc = Feature::getRegionShapeD( this->frame );

				long x = 0;

				for( long p = 0; p < ART_ANGULAR; p++ )
				{
					for( long r = 0; r < ART_RADIAL; r++ )
					{
						descriptors.at<float>( y, x++ ) = float( this->desc->GetRealValue( p, r ) );
					}
				}
			}
		}
	};
}

#endif
