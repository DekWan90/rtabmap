/*
Copyright (c) 2010-2014, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the Universite de Sherbrooke nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef KEYPOINTDESCRIPTOR_H_
#define KEYPOINTDESCRIPTOR_H_

#include "rtabmap/core/RtabmapExp.h" // DLL export/import defines

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <list>
#include <memory>
#include <iostream>
#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/SiftDescriptor.hpp"

namespace cv{
	class SURF;
	class SIFT;
	namespace gpu {
		class SURF_GPU;
		class ORB_GPU;
		class FAST_GPU;
	}
}

namespace rtabmap
{

	// Feature2D
	class RTABMAP_EXP Feature2D
	{
	public:
		enum Type
		{
			kFeatureUndef = -1,
			kFeatureSurf = 0,
			kFeatureSift = 1,
			kFeatureOrb = 2,
			kFeatureFastFreak = 3,
			kFeatureFastBrief = 4,
			kFeatureGfttFreak = 5,
			kFeatureGfttBrief = 6,
			kFeatureBrisk = 7,
			kFeatureMix = 8,
		};

		static Feature2D * create(Feature2D::Type & type, const ParametersMap & parameters);

		static void filterKeypointsByDepth(
			std::vector<cv::KeyPoint> & keypoints,
			const cv::Mat & depth,
			float maxDepth
		);

		static void filterKeypointsByDepth(
			std::vector<cv::KeyPoint> & keypoints,
			cv::Mat & descriptors,
			const cv::Mat & depth,
			float maxDepth
		);

		static void filterKeypointsByDisparity(
			std::vector<cv::KeyPoint> & keypoints,
			const cv::Mat & disparity,
			float minDisparity
		);

		static void filterKeypointsByDisparity(
			std::vector<cv::KeyPoint> & keypoints,
			cv::Mat & descriptors,
			const cv::Mat & disparity,
			float minDisparity
		);

		static void limitKeypoints(std::vector<cv::KeyPoint> & keypoints, int maxKeypoints);
		static void limitKeypoints(std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors, int maxKeypoints);

		static cv::Rect computeRoi(const cv::Mat & image, const std::string & roiRatios);
		static cv::Rect computeRoi(const cv::Mat & image, const std::vector<float> & roiRatios);

	public:
		virtual ~Feature2D() {}

		std::vector<cv::KeyPoint> generateKeypoints(const cv::Mat & image, int maxKeypoints=0, const cv::Rect & roi = cv::Rect()) const;
		cv::Mat generateDescriptors(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

		virtual void parseParameters(const ParametersMap & parameters) {}
		virtual Feature2D::Type getType() const = 0;

	protected:
		Feature2D(const ParametersMap & parameters = ParametersMap()) {}

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const = 0;
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const = 0;
	};

	//SURF
	class RTABMAP_EXP SURF : public Feature2D
	{
	public:
		SURF(const ParametersMap & parameters = ParametersMap());
		virtual ~SURF();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureSurf;}

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		double hessianThreshold_;
		int nOctaves_;
		int nOctaveLayers_;
		bool extended_;
		bool upright_;
		float gpuKeypointsRatio_;
		bool gpuVersion_;

		cv::SURF * _surf;
		cv::gpu::SURF_GPU * _gpuSurf;
	};

	//SIFT
	class RTABMAP_EXP SIFT : public Feature2D
	{
	public:
		SIFT(const ParametersMap & parameters = ParametersMap());
		virtual ~SIFT();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureSift;}

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		int nfeatures_;
		int nOctaveLayers_;
		double contrastThreshold_;
		double edgeThreshold_;
		double sigma_;

		cv::SIFT * _sift;
	};

	//ORB
	class RTABMAP_EXP ORB : public Feature2D
	{
	public:
		ORB(const ParametersMap & parameters = ParametersMap());
		virtual ~ORB();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureOrb;}

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		int nFeatures_;
		float scaleFactor_;
		int nLevels_;
		int edgeThreshold_;
		int firstLevel_;
		int WTA_K_;
		int scoreType_;
		int patchSize_;
		bool gpu_;

		int fastThreshold_;
		bool nonmaxSuppresion_;

		cv::ORB * _orb;
		cv::gpu::ORB_GPU * _gpuOrb;
	};

	//FAST
	class RTABMAP_EXP FAST : public Feature2D
	{
	public:
		FAST(const ParametersMap & parameters = ParametersMap());
		virtual ~FAST();

		virtual void parseParameters(const ParametersMap & parameters);

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;

	private:
		int threshold_;
		bool nonmaxSuppression_;
		bool gpu_;
		double gpuKeypointsRatio_;

		cv::FastFeatureDetector * _fast;
		cv::gpu::FAST_GPU * _gpuFast;
	};

	//FAST_BRIEF
	class RTABMAP_EXP FAST_BRIEF : public FAST
	{
	public:
		FAST_BRIEF(const ParametersMap & parameters = ParametersMap());
		virtual ~FAST_BRIEF();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureFastBrief;}

	private:
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		int bytes_;

		cv::BriefDescriptorExtractor * _brief;
	};

	//FAST_FREAK
	class RTABMAP_EXP FAST_FREAK : public FAST
	{
	public:
		FAST_FREAK(const ParametersMap & parameters = ParametersMap());
		virtual ~FAST_FREAK();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureFastFreak;}

	private:
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		bool orientationNormalized_;
		bool scaleNormalized_;
		float patternScale_;
		int nOctaves_;

		cv::FREAK * _freak;
	};

	//GFTT
	class RTABMAP_EXP GFTT : public Feature2D
	{
	public:
		GFTT(const ParametersMap & parameters = ParametersMap());
		virtual ~GFTT();

		virtual void parseParameters(const ParametersMap & parameters);

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;

	private:
		int _maxCorners;
		double _qualityLevel;
		double _minDistance;
		int _blockSize;
		bool _useHarrisDetector;
		double _k;

		cv::GFTTDetector * _gftt;
	};

	//GFTT_BRIEF
	class RTABMAP_EXP GFTT_BRIEF : public GFTT
	{
	public:
		GFTT_BRIEF(const ParametersMap & parameters = ParametersMap());
		virtual ~GFTT_BRIEF();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureGfttBrief;}

	private:
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		int bytes_;

		cv::BriefDescriptorExtractor * _brief;
	};

	//GFTT_FREAK
	class RTABMAP_EXP GFTT_FREAK : public GFTT
	{
	public:
		GFTT_FREAK(const ParametersMap & parameters = ParametersMap());
		virtual ~GFTT_FREAK();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureGfttFreak;}

	private:
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		bool orientationNormalized_;
		bool scaleNormalized_;
		float patternScale_;
		int nOctaves_;

		cv::FREAK * _freak;
	};

	//BRISK
	class RTABMAP_EXP BRISK : public Feature2D
	{
	public:
		BRISK(const ParametersMap & parameters = ParametersMap());
		virtual ~BRISK();

		virtual void parseParameters(const ParametersMap & parameters);
		virtual Feature2D::Type getType() const {return kFeatureBrisk;}

	private:
		virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const;
		virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	private:
		int thresh_;
		int octaves_;
		float patternScale_;

		cv::BRISK * brisk_;
	};

	class FixedPartition
	{
		public: FixedPartition( const long nFeatures, const double radius = 0.0, const bool overlapse = false );
		public: virtual ~FixedPartition(){}
		public: void detect( const cv::Mat image, std::vector<cv::KeyPoint> &keypoints );

		private: long nFeatures = 300;
		private: double radius = 10;
		private: bool overlapse = false;
	};

	// MIX
	class MIX : public Feature2D
	{
		public: MIX( const ParametersMap &parameters = ParametersMap() );
		public: virtual ~MIX();
		public: virtual Feature2D::Type getType() const { return kFeatureMix; }
		public: virtual void parseParameters( const ParametersMap &parameters );

		private: virtual std::vector<cv::KeyPoint> generateKeypointsImpl( const cv::Mat &image, const cv::Rect &roi ) const;
		private: virtual cv::Mat generateDescriptorsImpl( const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints ) const;

		// WRT-Map
		private: int titikUtama = Parameters::defaultCiriTitikUtama();
		private: int diskriptor = Parameters::defaultCiriDiskriptor();
		// SURF
		private: double hessianThreshold = Parameters::defaultCiriHessianThreshold();
		private: int nOctaves = Parameters::defaultCiriNOctaves();
		private: int nOctaveLayers = Parameters::defaultCiriNOctaveLayers();
		private: bool extended = Parameters::defaultCiriExtended();
		private: bool upright = Parameters::defaultCiriUpright();
		// SIFT
		private: int nFeatures = Parameters::defaultCiriNFeatures();
		private: double contrastThreshold = Parameters::defaultCiriContrastThreshold();
		private: double edgeThreshold = Parameters::defaultCiriEdgeThreshold();
		private: double sigma = Parameters::defaultCiriSigma();
		// ORB
		private: double scaleFactor = Parameters::defaultCiriScaleFactor();
		private: int nlevels = Parameters::defaultCiriNLevels();
		private: int firstLevel = Parameters::defaultCiriFirstLevel();
		private: int wta_k = Parameters::defaultCiriWTA_K();
		private: int scoreType = Parameters::defaultCiriScoreType();
		private: int patchSize = Parameters::defaultCiriPatchSize();
		// FAST
		private: int threshold = Parameters::defaultCiriThreshold();
		private: bool nonmaxSuppression = Parameters::defaultCiriNonMaxSuppression();
		// FASTX
		private: int type = Parameters::defaultCiriType();
		// FREAK
		private: bool orientationNormalized = Parameters::defaultCiriOrientationNormalized();
		private: bool scaleNormalized = Parameters::defaultCiriScaleNormalized();
		private: double patternScale = Parameters::defaultCiriPatternScale();
		// BRIEF
		private: int bytes = Parameters::defaultCiriBytes();
		// GFTT
		private: int maxCorners = Parameters::defaultCiriMaxCorners();
		private: double qualityLevel = Parameters::defaultCiriQualityLevel();
		private: double minDistance = Parameters::defaultCiriMinDistance();
		private: int blockSize = Parameters::defaultCiriBlockSize();
		private: bool useHarrisDetector = Parameters::defaultCiriUseHarrisDetector();
		private: double k = Parameters::defaultCiriK();
		// MSER
		private: int delta = Parameters::defaultCiriDelta();
		private: int minArea = Parameters::defaultCiriMinArea();
		private: int maxArea = Parameters::defaultCiriMaxArea();
		private: double maxVariation = Parameters::defaultCiriMaxVariation();
		private: double minDiversity = Parameters::defaultCiriMinDiversity();
		private: int maxEvolution = Parameters::defaultCiriMaxEvolution();
		private: double areaThreshold = Parameters::defaultCiriAreaThreshold();
		private: double minMargin = Parameters::defaultCiriMinMargin();
		private: int edgeBlurSize = Parameters::defaultCiriEdgeBlurSize();
		private: int radius = Parameters::defaultCiriRadius();
		// STAR
		private: int maxSize = Parameters::defaultCiriMaxSize();
		private: int responseThreshold = Parameters::defaultCiriResponseThreshold();
		private: int lineThresholdProjected = Parameters::defaultCiriLineThresholdProjected();
		private: int lineThresholdBinarized = Parameters::defaultCiriLineThresholdBinarized();
		private: int suppressNonmaxSize = Parameters::defaultCiriSuppressNonmaxSize();
		// DENSE
		private: double initFeatureScale = Parameters::defaultCiriInitFeatureScale();
		private: int featureScaleLevels = Parameters::defaultCiriFeatureScaleLevels();
		private: double featureScaleMul = Parameters::defaultCiriFeatureScaleMul();
		private: int initXyStep = Parameters::defaultCiriInitXyStep();
		private: int initImgBound = Parameters::defaultCiriInitImgBound();
		private: bool varyXyStepWithScale = Parameters::defaultCiriVaryXyStepWithScale();
		private: bool varyImgBoundWithScale = Parameters::defaultCiriVaryImgBoundWithScale();
		// SimpleBlobDetector
		private: cv::SimpleBlobDetector::Params sbdp;
		// FIXED_PARTITION
		private: bool overlapse = Parameters::defaultCiriOverlapse();
		// SIFTDESC
		private: int dims = Parameters::defaultCiriDims();
		private: int bins = Parameters::defaultCiriBins();
		private: double orientation = Parameters::defaultCiriOrientation();
		// GridAdaptedFeatureDetector
		private: int detector = Parameters::defaultCiriDetector();
		private: int gridRows = Parameters::defaultCiriGridRows();
		private: int gridCols = Parameters::defaultCiriGridCols();
		// OCDE
		private: int extractor = Parameters::defaultCiriExtractor();

		private: std::shared_ptr<cv::SURF> surf;
		private: std::shared_ptr<cv::SIFT> sift;
		private: std::shared_ptr<cv::FastFeatureDetector> fast;
		private: std::shared_ptr<cv::MSER> mser;
		private: std::shared_ptr<cv::ORB> orb;
		private: std::shared_ptr<cv::BRISK> brisk;
		private: std::shared_ptr<cv::FREAK> freak;
		private: std::shared_ptr<cv::StarFeatureDetector> star;
		private: std::shared_ptr<cv::GoodFeaturesToTrackDetector> gftt;
		private: std::shared_ptr<cv::DenseFeatureDetector> dense;
		private: std::shared_ptr<cv::SimpleBlobDetector> sbd;
		private: std::shared_ptr<FixedPartition> fpartition;
		private: std::shared_ptr<SiftDescriptor> siftdesc;
		private: std::shared_ptr<cv::GridAdaptedFeatureDetector> gafd;
		private: std::shared_ptr<cv::PyramidAdaptedFeatureDetector> pafd;
		private: std::shared_ptr<cv::OpponentColorDescriptorExtractor> ocde;
	};
}

#endif /* KEYPOINTDESCRIPTOR_H_ */
