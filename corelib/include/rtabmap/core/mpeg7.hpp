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

		protected: bool imgFlag = true;
		protected: bool grayFlag = true;
		protected: bool maskFlag = false;

		protected: int descSize = 64;
		protected: int numCoeff = 256;
		protected: int bitPlanesDiscarded = 0;

		protected: cv::Mat image;

		public: MPEG7(){}
		public: virtual ~MPEG7(){}

		protected: cv::Mat CropKeypoints( const cv::Mat image, const cv::KeyPoint keypoint ) const
		{
			double radius = keypoint.size / 2.0;
			return image( cv::Rect( keypoint.pt.x - radius, keypoint.pt.y - radius, keypoint.size, keypoint.size  ) );
		}
	};

	class ColorStructureDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ColorStructureDescriptor> csd;

		public: ColorStructureDescriptor( const int descSize = 64 )
		{
			this->descSize = descSize;
		}

		public: virtual ~ColorStructureDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors.create( keypoints.size(), this->descSize, CV_32FC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->csd = Feature::getColorStructureD( this->frame, this->descSize );

				for( unsigned long x = 0; x < this->csd->GetSize(); x++ )
				{
					descriptors.at<float>( y, x ) = float( this->csd->GetElement( x ) );
				}
			}
		}
	};

	class ScalableColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ScalableColorDescriptor> scd;

		public: ScalableColorDescriptor( const bool maskFlag = true, const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			this->maskFlag = maskFlag;
			this->numCoeff = numCoeff;
			this->bitPlanesDiscarded = bitPlanesDiscarded;
		}

		public: virtual ~ScalableColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors.create( keypoints.size(), this->numCoeff, CV_32FC1 );

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				this->image = CropKeypoints( image, keypoints[y] );
				this->frame.reset( new Frame( this->image, this->imgFlag, this->grayFlag, this->maskFlag ) );
				this->scd = Feature::getScalableColorD( this->frame, this->maskFlag, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->scd->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<float>( y, x ) = float( this->scd->GetCoeffSign( x ) * this->scd->GetCoefficient( x ) );
				}
			}
		}
	};

	class GoFGoPColorDescriptor : public MPEG7
	{
		private: std::shared_ptr<XM::ScalableColorDescriptor> scd;

		public: GoFGoPColorDescriptor( const int numCoeff = 256, const int bitPlanesDiscarded = 0 )
		{
			this->numCoeff = numCoeff;
			this->bitPlanesDiscarded = bitPlanesDiscarded;
		}

		public: virtual ~GoFGoPColorDescriptor(){}

		public: void compute( const cv::Mat image, const std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors )
		{
			descriptors.create( keypoints.size(), this->numCoeff, CV_32FC1 );
			std::vector<cv::Mat> vImage;

			for( unsigned long y = 0; y < keypoints.size(); y++ )
			{
				vImage.push_back( CropKeypoints( image, keypoints[y] ) );
				this->scd = Feature::getGoFColorD( vImage, this->numCoeff, this->bitPlanesDiscarded );

				for( unsigned long x = 0; x < this->scd->GetNumberOfCoefficients(); x++ )
				{
					descriptors.at<float>( y, x ) = float( this->scd->GetCoeffSign( x ) * this->scd->GetCoefficient( x ) );
				}
			}
		}
	};
}

#endif
