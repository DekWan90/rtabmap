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
		protected: bool maskFlag = false;

		// Color Structure Descriptor
		protected: int descSize = 64;

		// Scalable Color Descriptor
		protected: int numCoeff = 256;
		protected: int bitPlanesDiscarded = 0;

		// GoF GoP Color Descriptor

		// Dominant Color Descriptor
		protected: bool normalize = true;
		protected: bool variance = true;
		protected: bool spatial = true;
		protected: int bin1 = 32;
		protected: int bin2 = 32;
		protected: int bin3 = 32;

		// Color Layout Descriptor
		protected: int numberOfYCoeff = 64;
		protected: int numberOfCCoeff = 28;

		// Edge Histogram Descriptor

		// Homogeneous Texture Descriptor
		protected: bool layerFlag = true;

		public: MPEG7(){}
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

		public: ColorStructureDescriptor( const int descSize = 64 )
		{
			this->descSize = descSize;
		}

		public: virtual ~ColorStructureDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), this->descSize, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
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

		public: ScalableColorDescriptor( const bool maskFlag = true, const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			this->maskFlag = maskFlag;
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
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->desc = Feature::getScalableColorD( this->frame, this->maskFlag, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->desc->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<uchar>( y, x ) = uchar( this->desc->GetCoeffSign( x ) * this->desc->GetCoefficient( x ) );
				}
			}
		}
	};

	class GoFGoPColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ScalableColorDescriptor> desc;

		public: GoFGoPColorDescriptor( const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			this->numCoeff = numCoeff;
			this->bitPlanesDiscarded = bitPlanesDiscarded;
		}

		public: virtual ~GoFGoPColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors = cv::Mat::zeros( keypoints.size(), this->numCoeff, CV_8UC1 );
			std::vector<cv::Mat> vImage;

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				vImage.push_back( CropKeypoints( image, keypoints[y] ) );
				this->desc = Feature::getGoFColorD( vImage, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->desc->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<uchar>( y, x ) = uchar( this->desc->GetCoeffSign( x ) * this->desc->GetCoefficient( x ) );
				}
			}
		}
	};

	class DominantColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::DominantColorDescriptor> desc;

		public: DominantColorDescriptor( const bool normalize = true, const bool variance = true, const bool spatial = true, const int bin1 = 32, const int bin2 = 32, const int bin3 = 32 )
		{
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

			descriptors = cv::Mat::zeros( keypoints.size(), ( 8 * 7 ) + 2, CV_8UC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->desc = Feature::getDominantColorD( this->frame, this->normalize, this->variance, this->spatial, this->bin1, this->bin2, this->bin3 );

				long x = 0;
				descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetSpatialCoherency() );
				descriptors.at<uchar>( y, x++ ) = uchar( this->desc->GetDominantColorsNumber() );

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

		public: ColorLayoutDescriptor( int numberOfYCoeff = 64, int numberOfCCoeff = 28 )
		{
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
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
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

		public: EdgeHistogramDescriptor(){}
		public: virtual ~EdgeHistogramDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{

			descriptors = cv::Mat::zeros( keypoints.size(), 80, CV_32FC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->desc = Feature::getEdgeHistogramD( this->frame );

				for( unsigned long x = 0; x < this->desc->GetSize(); x++ )
				{
					descriptors.at<float>( y, x ) = float( this->desc->GetEdgeHistogramD()[x] );
				}
			}
		}
	};

	class HomogeneousTextureDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::HomogeneousTextureDescriptor> desc;

		public: HomogeneousTextureDescriptor( bool layerFlag = true )
		{
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

				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->frame->setGray( this->image );

				this->desc = Feature::getHomogeneousTextureD( this->frame, this->layerFlag );

				for( unsigned long x = 0; x < 62; x++ )
				{
					descriptors.at<uchar>( y, x ) = uchar( this->desc->GetHomogeneousTextureFeature()[x] );
				}
			}
		}
	};
}

#endif
