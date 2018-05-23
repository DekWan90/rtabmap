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

#include "rtabmap/core/Features2d.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/utilite/UStl.h"
#include "rtabmap/utilite/UConversion.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UTimer.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/version.hpp>

#ifdef WITH_NONFREE
#if CV_MAJOR_VERSION > 2 || (CV_MAJOR_VERSION >=2 && CV_MINOR_VERSION >=4)
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>
#endif
#endif

namespace rtabmap
{
	void Feature2D::filterKeypointsByDepth(
		std::vector<cv::KeyPoint> & keypoints,
		const cv::Mat & depth,
		float maxDepth
	)
	{
		cv::Mat descriptors;
		filterKeypointsByDepth(keypoints, descriptors, depth, maxDepth);
	}

	void Feature2D::filterKeypointsByDepth(
		std::vector<cv::KeyPoint> & keypoints,
		cv::Mat & descriptors,
		const cv::Mat & depth,
		float maxDepth
	)
	{
		if(!depth.empty() && maxDepth > 0.0f && (descriptors.empty() || descriptors.rows == (int)keypoints.size()))
		{
			std::vector<cv::KeyPoint> output(keypoints.size());
			std::vector<int> indexes(keypoints.size(), 0);
			int oi=0;
			bool isInMM = depth.type() == CV_16UC1;
			for(unsigned int i=0; i<keypoints.size(); ++i)
			{
				int u = int(keypoints[i].pt.x+0.5f);
				int v = int(keypoints[i].pt.y+0.5f);
				if(u >=0 && u<depth.cols && v >=0 && v<depth.rows)
				{
					float d = isInMM?(float)depth.at<uint16_t>(v,u)*0.001f:depth.at<float>(v,u);
					if(d!=0.0f && uIsFinite(d) && d < maxDepth)
					{
						output[oi++] = keypoints[i];
						indexes[i] = 1;
					}
				}
			}
			output.resize(oi);
			keypoints = output;

			if(!descriptors.empty() && (int)keypoints.size() != descriptors.rows)
			{
				if(keypoints.size() == 0)
				{
					descriptors = cv::Mat();
				}
				else
				{
					cv::Mat newDescriptors((int)keypoints.size(), descriptors.cols, descriptors.type());
					int di = 0;
					for(unsigned int i=0; i<indexes.size(); ++i)
					{
						if(indexes[i] == 1)
						{
							if(descriptors.type() == CV_32FC1)
							{
								memcpy(newDescriptors.ptr<float>(di++), descriptors.ptr<float>(i), descriptors.cols*sizeof(float));
							}
							else // CV_8UC1
							{
								memcpy(newDescriptors.ptr<char>(di++), descriptors.ptr<char>(i), descriptors.cols*sizeof(char));
							}
						}
					}
					descriptors = newDescriptors;
				}
			}
		}
	}

	void Feature2D::filterKeypointsByDisparity(
		std::vector<cv::KeyPoint> & keypoints,
		const cv::Mat & disparity,
		float minDisparity
	)
	{
		cv::Mat descriptors;
		filterKeypointsByDisparity(keypoints, descriptors, disparity, minDisparity);
	}

	void Feature2D::filterKeypointsByDisparity(
		std::vector<cv::KeyPoint> & keypoints,
		cv::Mat & descriptors,
		const cv::Mat & disparity,
		float minDisparity
	)
	{
		if(!disparity.empty() && minDisparity > 0.0f && (descriptors.empty() || descriptors.rows == (int)keypoints.size()))
		{
			std::vector<cv::KeyPoint> output(keypoints.size());
			std::vector<int> indexes(keypoints.size(), 0);
			int oi=0;
			for(unsigned int i=0; i<keypoints.size(); ++i)
			{
				int u = int(keypoints[i].pt.x+0.5f);
				int v = int(keypoints[i].pt.y+0.5f);
				if(u >=0 && u<disparity.cols && v >=0 && v<disparity.rows)
				{
					float d = disparity.type() == CV_16SC1?float(disparity.at<short>(v,u))/16.0f:disparity.at<float>(v,u);
					if(d!=0.0f && uIsFinite(d) && d >= minDisparity)
					{
						output[oi++] = keypoints[i];
						indexes[i] = 1;
					}
				}
			}
			output.resize(oi);
			keypoints = output;

			if(!descriptors.empty() && (int)keypoints.size() != descriptors.rows)
			{
				if(keypoints.size() == 0)
				{
					descriptors = cv::Mat();
				}
				else
				{
					cv::Mat newDescriptors((int)keypoints.size(), descriptors.cols, descriptors.type());
					int di = 0;
					for(unsigned int i=0; i<indexes.size(); ++i)
					{
						if(indexes[i] == 1)
						{
							if(descriptors.type() == CV_32FC1)
							{
								memcpy(newDescriptors.ptr<float>(di++), descriptors.ptr<float>(i), descriptors.cols*sizeof(float));
							}
							else // CV_8UC1
							{
								memcpy(newDescriptors.ptr<char>(di++), descriptors.ptr<char>(i), descriptors.cols*sizeof(char));
							}
						}
					}
					descriptors = newDescriptors;
				}
			}
		}
	}

	void Feature2D::limitKeypoints(std::vector<cv::KeyPoint> & keypoints, int maxKeypoints)
	{
		cv::Mat descriptors;
		limitKeypoints(keypoints, descriptors, maxKeypoints);
	}

	void Feature2D::limitKeypoints(std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors, int maxKeypoints)
	{
		UASSERT_MSG((int)keypoints.size() == descriptors.rows || descriptors.rows == 0, uFormat("keypoints=%d descriptors=%d", (int)keypoints.size(), descriptors.rows).c_str());
		if(maxKeypoints > 0 && (int)keypoints.size() > maxKeypoints)
		{
			UTimer timer;
			ULOGGER_DEBUG("too much words (%d), removing words with the hessian threshold", keypoints.size());
			// Remove words under the new hessian threshold

			// Sort words by hessian
			std::multimap<float, int> hessianMap; // <hessian,id>
			for(unsigned int i = 0; i <keypoints.size(); ++i)
			{
				//Keep track of the data, to be easier to manage the data in the next step
				hessianMap.insert(std::pair<float, int>(fabs(keypoints[i].response), i));
			}

			// Remove them from the signature
			int removed = (int)hessianMap.size()-maxKeypoints;
			std::multimap<float, int>::reverse_iterator iter = hessianMap.rbegin();
			std::vector<cv::KeyPoint> kptsTmp(maxKeypoints);
			cv::Mat descriptorsTmp;
			if(descriptors.rows)
			{
				descriptorsTmp = cv::Mat(maxKeypoints, descriptors.cols, descriptors.type());
			}
			for(unsigned int k=0; k < kptsTmp.size() && iter!=hessianMap.rend(); ++k, ++iter)
			{
				kptsTmp[k] = keypoints[iter->second];
				if(descriptors.rows)
				{
					if(descriptors.type() == CV_32FC1)
					{
						memcpy(descriptorsTmp.ptr<float>(k), descriptors.ptr<float>(iter->second), descriptors.cols*sizeof(float));
					}
					else
					{
						memcpy(descriptorsTmp.ptr<char>(k), descriptors.ptr<char>(iter->second), descriptors.cols*sizeof(char));
					}
				}
			}
			ULOGGER_DEBUG("%d keypoints removed, (kept %d), minimum response=%f", removed, (int)keypoints.size(), kptsTmp.size()?kptsTmp.back().response:0.0f);
			ULOGGER_DEBUG("removing words time = %f s", timer.ticks());
			keypoints = kptsTmp;
			if(descriptors.rows)
			{
				descriptors = descriptorsTmp;
			}
		}
	}

	cv::Rect Feature2D::computeRoi(const cv::Mat & image, const std::string & roiRatios)
	{
		std::list<std::string> strValues = uSplit(roiRatios, ' ');
		if(strValues.size() != 4)
		{
			UERROR("The number of values must be 4 (roi=\"%s\")", roiRatios.c_str());
		}
		else
		{
			std::vector<float> values(4);
			unsigned int i=0;
			for(std::list<std::string>::iterator iter = strValues.begin(); iter!=strValues.end(); ++iter)
			{
				values[i] = std::atof((*iter).c_str());
				++i;
			}

			if(values[0] >= 0 && values[0] < 1 && values[0] < 1.0f-values[1] &&
				values[1] >= 0 && values[1] < 1 && values[1] < 1.0f-values[0] &&
				values[2] >= 0 && values[2] < 1 && values[2] < 1.0f-values[3] &&
				values[3] >= 0 && values[3] < 1 && values[3] < 1.0f-values[2]
			)
			{
				return computeRoi(image, values);
			}
			else
			{
				UERROR("The roi ratios are not valid (roi=\"%s\")", roiRatios.c_str());
			}
		}
		return cv::Rect();
	}

	cv::Rect Feature2D::computeRoi(const cv::Mat & image, const std::vector<float> & roiRatios)
	{
		if(!image.empty() && roiRatios.size() == 4)
		{
			float width = image.cols;
			float height = image.rows;
			cv::Rect roi(0, 0, width, height);
			UDEBUG("roi ratios = %f, %f, %f, %f", roiRatios[0],roiRatios[1],roiRatios[2],roiRatios[3]);
			UDEBUG("roi = %d, %d, %d, %d", roi.x, roi.y, roi.width, roi.height);

			//left roi
			if(roiRatios[0] > 0 && roiRatios[0] < 1 - roiRatios[1])
			{
				roi.x = width * roiRatios[0];
			}

			//right roi
			roi.width = width - roi.x;
			if(roiRatios[1] > 0 && roiRatios[1] < 1 - roiRatios[0])
			{
				roi.width -= width * roiRatios[1];
			}

			//top roi
			if(roiRatios[2] > 0 && roiRatios[2] < 1 - roiRatios[3])
			{
				roi.y = height * roiRatios[2];
			}

			//bottom roi
			roi.height = height - roi.y;
			if(roiRatios[3] > 0 && roiRatios[3] < 1 - roiRatios[2])
			{
				roi.height -= height * roiRatios[3];
			}
			UDEBUG("roi = %d, %d, %d, %d", roi.x, roi.y, roi.width, roi.height);

			return roi;
		}
		else
		{
			UERROR("Image is null or _roiRatios(=%d) != 4", roiRatios.size());
			return cv::Rect();
		}
	}

	/////////////////////
	// Feature2D
	/////////////////////
	Feature2D * Feature2D::create(Feature2D::Type & type, const ParametersMap & parameters)
	{
		if(RTABMAP_NONFREE == 0 &&
			(type == Feature2D::kFeatureSurf || type == Feature2D::kFeatureSift)
		)
		{
			UERROR("SURF/SIFT features cannot be used because OpenCV was not built with nonfree module. ORB is used instead.");
			type = Feature2D::kFeatureOrb;
		}

		Feature2D * feature2D = 0;

		switch(type)
		{
			case Feature2D::kFeatureSurf:
			feature2D = new SURF(parameters);
			break;
			case Feature2D::kFeatureSift:
			feature2D = new SIFT(parameters);
			break;
			case Feature2D::kFeatureOrb:
			feature2D = new ORB(parameters);
			break;
			case Feature2D::kFeatureFastBrief:
			feature2D = new FAST_BRIEF(parameters);
			break;
			case Feature2D::kFeatureFastFreak:
			feature2D = new FAST_FREAK(parameters);
			break;
			case Feature2D::kFeatureGfttFreak:
			feature2D = new GFTT_FREAK(parameters);
			break;
			case Feature2D::kFeatureGfttBrief:
			feature2D = new GFTT_BRIEF(parameters);
			break;
			case Feature2D::kFeatureBrisk:
			feature2D = new BRISK(parameters);
			break;
			case Feature2D::kFeatureDekWan:
			feature2D = new DekWan( parameters );
			break;
			#ifdef WITH_NONFREE
			default:
			feature2D = new SURF(parameters);
			type = Feature2D::kFeatureSurf;
			break;
			#else
			default:
			feature2D = new ORB(parameters);
			type = Feature2D::kFeatureOrb;
			break;
			#endif

		}
		return feature2D;
	}

	std::vector<cv::KeyPoint> Feature2D::generateKeypoints(const cv::Mat & image, int maxKeypoints, const cv::Rect & roi) const
	{
		ULOGGER_DEBUG("");

		std::vector<cv::KeyPoint> keypoints;
		if(!image.empty() /* && image.channels() == 1 && image.type() == CV_8U*/)
		{
			UTimer timer;

			// Get keypoints
			keypoints = this->generateKeypointsImpl(image, roi.width && roi.height?roi:cv::Rect(0,0,image.cols, image.rows));
			ULOGGER_DEBUG("Keypoints extraction time = %f s, keypoints extracted = %d", timer.ticks(), keypoints.size());

			limitKeypoints(keypoints, maxKeypoints);

			if(roi.x || roi.y)
			{
				// Adjust keypoint position to raw image
				for(std::vector<cv::KeyPoint>::iterator iter=keypoints.begin(); iter!=keypoints.end(); ++iter)
				{
					iter->pt.x += roi.x;
					iter->pt.y += roi.y;
				}
			}
		}
		else if(image.empty())
		{
			UERROR("Image is null!");
		}
		else
		{
			UERROR("Image format must be mono8. Current has %d channels and type = %d, size=%d,%d",
			image.channels(), image.type(), image.cols, image.rows);
		}

		return keypoints;
	}

	cv::Mat Feature2D::generateDescriptors(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		cv::Mat descriptors = generateDescriptorsImpl(image, keypoints);
		UASSERT_MSG(descriptors.rows == (int)keypoints.size(), uFormat("descriptors=%d, keypoints=%d", descriptors.rows, (int)keypoints.size()).c_str());
		UDEBUG("Descriptors extracted = %d, remaining kpts=%d", descriptors.rows, (int)keypoints.size());
		return descriptors;
	}

	//////////////////////////
	//SURF
	//////////////////////////
	SURF::SURF(const ParametersMap & parameters) :
	hessianThreshold_(Parameters::defaultSURFHessianThreshold()),
	nOctaves_(Parameters::defaultSURFNOctaves()),
	nOctaveLayers_(Parameters::defaultSURFNOctaveLayers()),
	extended_(Parameters::defaultSURFExtended()),
	upright_(Parameters::defaultSURFUpright()),
	gpuKeypointsRatio_(Parameters::defaultSURFGpuKeypointsRatio()),
	gpuVersion_(Parameters::defaultSURFGpuVersion()),
	_surf(0),
	_gpuSurf(0)
	{
		parseParameters(parameters);
	}

	SURF::~SURF()
	{
		#ifdef WITH_NONFREE
		if(_surf)
		{
			delete _surf;
		}
		if(_gpuSurf)
		{
			delete _gpuSurf;
		}
		#endif
	}

	void SURF::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kSURFExtended(), extended_);
		Parameters::parse(parameters, Parameters::kSURFHessianThreshold(), hessianThreshold_);
		Parameters::parse(parameters, Parameters::kSURFNOctaveLayers(), nOctaveLayers_);
		Parameters::parse(parameters, Parameters::kSURFNOctaves(), nOctaves_);
		Parameters::parse(parameters, Parameters::kSURFUpright(), upright_);
		Parameters::parse(parameters, Parameters::kSURFGpuKeypointsRatio(), gpuKeypointsRatio_);
		Parameters::parse(parameters, Parameters::kSURFGpuVersion(), gpuVersion_);

		#ifdef WITH_NONFREE
		if(_gpuSurf)
		{
			delete _gpuSurf;
			_gpuSurf = 0;
		}
		if(_surf)
		{
			delete _surf;
			_surf = 0;
		}

		if(gpuVersion_ && cv::gpu::getCudaEnabledDeviceCount())
		{
			_gpuSurf = new cv::gpu::SURF_GPU(hessianThreshold_, nOctaves_, nOctaveLayers_, extended_, gpuKeypointsRatio_, upright_);
		}
		else
		{
			if(gpuVersion_)
			{
				UWARN("GPU version of SURF not available! Using CPU version instead...");
			}

			_surf = new cv::SURF(hessianThreshold_, nOctaves_, nOctaveLayers_, extended_, upright_);
		}
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SURF cannot be used!");
		#endif
	}

	std::vector<cv::KeyPoint> SURF::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;

		#ifdef WITH_NONFREE
		cv::Mat imgRoi(image, roi);
		if(_gpuSurf)
		{
			cv::gpu::GpuMat imgGpu(imgRoi);
			(*_gpuSurf)(imgGpu, cv::gpu::GpuMat(), keypoints);
		}
		else
		{
			_surf->detect(imgRoi, keypoints);
		}
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SURF cannot be used!");
		#endif
		return keypoints;
	}

	cv::Mat SURF::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		#ifdef WITH_NONFREE
		if(_gpuSurf)
		{
			cv::gpu::GpuMat imgGpu(image);
			cv::gpu::GpuMat descriptorsGPU;
			(*_gpuSurf)(imgGpu, cv::gpu::GpuMat(), keypoints, descriptorsGPU, true);

			// Download descriptors
			if (descriptorsGPU.empty())
			descriptors = cv::Mat();
			else
			{
				UASSERT(descriptorsGPU.type() == CV_32F);
				descriptors = cv::Mat(descriptorsGPU.size(), CV_32F);
				descriptorsGPU.download(descriptors);
			}
		}
		else
		{
			_surf->compute(image, keypoints, descriptors);
		}
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SURF cannot be used!");
		#endif

		return descriptors;
	}

	//////////////////////////
	//SIFT
	//////////////////////////
	SIFT::SIFT(const ParametersMap & parameters) :
	nfeatures_(Parameters::defaultSIFTNFeatures()),
	nOctaveLayers_(Parameters::defaultSIFTNOctaveLayers()),
	contrastThreshold_(Parameters::defaultSIFTContrastThreshold()),
	edgeThreshold_(Parameters::defaultSIFTEdgeThreshold()),
	sigma_(Parameters::defaultSIFTSigma()),
	_sift(0)
	{
		parseParameters(parameters);
	}

	SIFT::~SIFT()
	{
		#ifdef WITH_NONFREE
		if(_sift)
		{
			delete _sift;
		}
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SIFT cannot be used!");
		#endif
	}

	void SIFT::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kSIFTContrastThreshold(), contrastThreshold_);
		Parameters::parse(parameters, Parameters::kSIFTEdgeThreshold(), edgeThreshold_);
		Parameters::parse(parameters, Parameters::kSIFTNFeatures(), nfeatures_);
		Parameters::parse(parameters, Parameters::kSIFTNOctaveLayers(), nOctaveLayers_);
		Parameters::parse(parameters, Parameters::kSIFTSigma(), sigma_);

		#ifdef WITH_NONFREE
		if(_sift)
		{
			delete _sift;
			_sift = 0;
		}

		_sift = new cv::SIFT(nfeatures_, nOctaveLayers_, contrastThreshold_, edgeThreshold_, sigma_);
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SIFT cannot be used!");
		#endif
	}

	std::vector<cv::KeyPoint> SIFT::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;
		#ifdef WITH_NONFREE
		cv::Mat imgRoi(image, roi);
		_sift->detect(imgRoi, keypoints); // Opencv keypoints
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SIFT cannot be used!");
		#endif
		return keypoints;
	}

	cv::Mat SIFT::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		#ifdef WITH_NONFREE
		_sift->compute(image, keypoints, descriptors);
		#else
		UERROR("RTAB-Map is not built with OpenCV nonfree module so SIFT cannot be used!");
		#endif
		return descriptors;
	}

	//////////////////////////
	//ORB
	//////////////////////////
	ORB::ORB(const ParametersMap & parameters) :
	nFeatures_(Parameters::defaultORBNFeatures()),
	scaleFactor_(Parameters::defaultORBScaleFactor()),
	nLevels_(Parameters::defaultORBNLevels()),
	edgeThreshold_(Parameters::defaultORBEdgeThreshold()),
	firstLevel_(Parameters::defaultORBFirstLevel()),
	WTA_K_(Parameters::defaultORBWTA_K()),
	scoreType_(Parameters::defaultORBScoreType()),
	patchSize_(Parameters::defaultORBPatchSize()),
	gpu_(Parameters::defaultORBGpu()),
	fastThreshold_(Parameters::defaultFASTThreshold()),
	nonmaxSuppresion_(Parameters::defaultFASTNonMaxSuppression()),
	_orb(0),
	_gpuOrb(0)
	{
		parseParameters(parameters);
	}

	ORB::~ORB()
	{
		if(_orb)
		{
			delete _orb;
		}
		if(_gpuOrb)
		{
			delete _gpuOrb;
		}
	}

	void ORB::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kORBNFeatures(), nFeatures_);
		Parameters::parse(parameters, Parameters::kORBScaleFactor(), scaleFactor_);
		Parameters::parse(parameters, Parameters::kORBNLevels(), nLevels_);
		Parameters::parse(parameters, Parameters::kORBEdgeThreshold(), edgeThreshold_);
		Parameters::parse(parameters, Parameters::kORBFirstLevel(), firstLevel_);
		Parameters::parse(parameters, Parameters::kORBWTA_K(), WTA_K_);
		Parameters::parse(parameters, Parameters::kORBScoreType(), scoreType_);
		Parameters::parse(parameters, Parameters::kORBPatchSize(), patchSize_);
		Parameters::parse(parameters, Parameters::kORBGpu(), gpu_);

		Parameters::parse(parameters, Parameters::kFASTThreshold(), fastThreshold_);
		Parameters::parse(parameters, Parameters::kFASTNonMaxSuppression(), nonmaxSuppresion_);

		if(_gpuOrb)
		{
			delete _gpuOrb;
			_gpuOrb = 0;
		}
		if(_orb)
		{
			delete _orb;
			_orb = 0;
		}

		if(gpu_ && cv::gpu::getCudaEnabledDeviceCount())
		{
			_gpuOrb = new cv::gpu::ORB_GPU(nFeatures_, scaleFactor_, nLevels_, edgeThreshold_, firstLevel_, WTA_K_, scoreType_, patchSize_);
			_gpuOrb->setFastParams(fastThreshold_, nonmaxSuppresion_);
		}
		else
		{
			if(gpu_)
			{
				UWARN("GPU version of ORB not available! Using CPU version instead...");
			}
			_orb = new cv::ORB(nFeatures_, scaleFactor_, nLevels_, edgeThreshold_, firstLevel_, WTA_K_, scoreType_, patchSize_);
		}
	}

	std::vector<cv::KeyPoint> ORB::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat imgRoi(image, roi);
		if(_gpuOrb)
		{
			cv::gpu::GpuMat imgGpu(imgRoi);
			(*_gpuOrb)(imgGpu, cv::gpu::GpuMat(), keypoints);
		}
		else
		{
			_orb->detect(imgRoi, keypoints);
		}

		return keypoints;
	}

	cv::Mat ORB::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		if(image.empty())
		{
			ULOGGER_ERROR("Image is null ?!?");
			return descriptors;
		}
		if(_gpuOrb)
		{
			cv::gpu::GpuMat imgGpu(image);
			cv::gpu::GpuMat descriptorsGPU;
			(*_gpuOrb)(imgGpu, cv::gpu::GpuMat(), keypoints, descriptorsGPU);

			// Download descriptors
			if (descriptorsGPU.empty())
			descriptors = cv::Mat();
			else
			{
				UASSERT(descriptorsGPU.type() == CV_32F);
				descriptors = cv::Mat(descriptorsGPU.size(), CV_32F);
				descriptorsGPU.download(descriptors);
			}
		}
		else
		{
			_orb->compute(image, keypoints, descriptors);
		}

		return descriptors;
	}

	//////////////////////////
	//FAST
	//////////////////////////
	FAST::FAST(const ParametersMap & parameters) :
	threshold_(Parameters::defaultFASTThreshold()),
	nonmaxSuppression_(Parameters::defaultFASTNonMaxSuppression()),
	gpu_(Parameters::defaultFASTGpu()),
	gpuKeypointsRatio_(Parameters::defaultFASTGpuKeypointsRatio()),
	_fast(0),
	_gpuFast(0)
	{
		parseParameters(parameters);
	}

	FAST::~FAST()
	{
		if(_fast)
		{
			delete _fast;
		}
		if(_gpuFast)
		{
			delete _gpuFast;
		}
	}

	void FAST::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kFASTThreshold(), threshold_);
		Parameters::parse(parameters, Parameters::kFASTNonMaxSuppression(), nonmaxSuppression_);
		Parameters::parse(parameters, Parameters::kFASTGpu(), gpu_);
		Parameters::parse(parameters, Parameters::kFASTGpuKeypointsRatio(), gpuKeypointsRatio_);

		if(_gpuFast)
		{
			delete _gpuFast;
			_gpuFast = 0;
		}
		if(_fast)
		{
			delete _fast;
			_fast = 0;
		}

		if(gpu_ && cv::gpu::getCudaEnabledDeviceCount())
		{
			_gpuFast = new cv::gpu::FAST_GPU(threshold_, nonmaxSuppression_, gpuKeypointsRatio_);
		}
		else
		{
			if(gpu_)
			{
				UWARN("GPU version of FAST not available! Using CPU version instead...");
			}
			_fast = new cv::FastFeatureDetector(threshold_, nonmaxSuppression_);
		}
	}

	std::vector<cv::KeyPoint> FAST::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat imgRoi(image, roi);
		if(_gpuFast)
		{
			cv::gpu::GpuMat imgGpu(imgRoi);
			(*_gpuFast)(imgGpu, cv::gpu::GpuMat(), keypoints);
		}
		else
		{
			_fast->detect(imgRoi, keypoints); // Opencv keypoints
		}
		return keypoints;
	}

	//////////////////////////
	//FAST-BRIEF
	//////////////////////////
	FAST_BRIEF::FAST_BRIEF(const ParametersMap & parameters) :
	FAST(parameters),
	bytes_(Parameters::defaultBRIEFBytes()),
	_brief(0)
	{
		parseParameters(parameters);
	}

	FAST_BRIEF::~FAST_BRIEF()
	{
		if(_brief)
		{
			delete _brief;
		}
	}

	void FAST_BRIEF::parseParameters(const ParametersMap & parameters)
	{
		FAST::parseParameters(parameters);

		Parameters::parse(parameters, Parameters::kBRIEFBytes(), bytes_);
		if(_brief)
		{
			delete _brief;
			_brief = 0;
		}
		_brief = new cv::BriefDescriptorExtractor(bytes_);
	}

	cv::Mat FAST_BRIEF::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		_brief->compute(image, keypoints, descriptors);
		return descriptors;
	}

	//////////////////////////
	//FAST-FREAK
	//////////////////////////
	FAST_FREAK::FAST_FREAK(const ParametersMap & parameters) :
	FAST(parameters),
	orientationNormalized_(Parameters::defaultFREAKOrientationNormalized()),
	scaleNormalized_(Parameters::defaultFREAKScaleNormalized()),
	patternScale_(Parameters::defaultFREAKPatternScale()),
	nOctaves_(Parameters::defaultFREAKNOctaves()),
	_freak(0)
	{
		parseParameters(parameters);
	}

	FAST_FREAK::~FAST_FREAK()
	{
		if(_freak)
		{
			delete _freak;
		}
	}

	void FAST_FREAK::parseParameters(const ParametersMap & parameters)
	{
		FAST::parseParameters(parameters);

		Parameters::parse(parameters, Parameters::kFREAKOrientationNormalized(), orientationNormalized_);
		Parameters::parse(parameters, Parameters::kFREAKScaleNormalized(), scaleNormalized_);
		Parameters::parse(parameters, Parameters::kFREAKPatternScale(), patternScale_);
		Parameters::parse(parameters, Parameters::kFREAKNOctaves(), nOctaves_);

		if(_freak)
		{
			delete _freak;
			_freak = 0;
		}

		_freak = new cv::FREAK(orientationNormalized_, scaleNormalized_, patternScale_, nOctaves_);
	}

	cv::Mat FAST_FREAK::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		_freak->compute(image, keypoints, descriptors);
		return descriptors;
	}

	//////////////////////////
	//GFTT
	//////////////////////////
	GFTT::GFTT(const ParametersMap & parameters) :
	_maxCorners(Parameters::defaultGFTTMaxCorners()),
	_qualityLevel(Parameters::defaultGFTTQualityLevel()),
	_minDistance(Parameters::defaultGFTTMinDistance()),
	_blockSize(Parameters::defaultGFTTBlockSize()),
	_useHarrisDetector(Parameters::defaultGFTTUseHarrisDetector()),
	_k(Parameters::defaultGFTTK()),
	_gftt(0)
	{
		parseParameters(parameters);
	}

	GFTT::~GFTT()
	{
		if(_gftt)
		{
			delete _gftt;
		}
	}

	void GFTT::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kGFTTMaxCorners(), _maxCorners);
		Parameters::parse(parameters, Parameters::kGFTTQualityLevel(), _qualityLevel);
		Parameters::parse(parameters, Parameters::kGFTTMinDistance(), _minDistance);
		Parameters::parse(parameters, Parameters::kGFTTBlockSize(), _blockSize);
		Parameters::parse(parameters, Parameters::kGFTTUseHarrisDetector(), _useHarrisDetector);
		Parameters::parse(parameters, Parameters::kGFTTK(), _k);

		if(_gftt)
		{
			delete _gftt;
			_gftt = 0;
		}
		_gftt = new cv::GFTTDetector(_maxCorners, _qualityLevel, _minDistance, _blockSize, _useHarrisDetector ,_k);
	}

	std::vector<cv::KeyPoint> GFTT::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat imgRoi(image, roi);
		_gftt->detect(imgRoi, keypoints); // Opencv keypoints
		return keypoints;
	}

	//////////////////////////
	//FAST-BRIEF
	//////////////////////////
	GFTT_BRIEF::GFTT_BRIEF(const ParametersMap & parameters) :
	GFTT(parameters),
	bytes_(Parameters::defaultBRIEFBytes()),
	_brief(0)
	{
		parseParameters(parameters);
	}

	GFTT_BRIEF::~GFTT_BRIEF()
	{
		if(_brief)
		{
			delete _brief;
		}
	}

	void GFTT_BRIEF::parseParameters(const ParametersMap & parameters)
	{
		GFTT::parseParameters(parameters);

		Parameters::parse(parameters, Parameters::kBRIEFBytes(), bytes_);
		if(_brief)
		{
			delete _brief;
			_brief = 0;
		}
		_brief = new cv::BriefDescriptorExtractor(bytes_);
	}

	cv::Mat GFTT_BRIEF::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		_brief->compute(image, keypoints, descriptors);
		return descriptors;
	}

	//////////////////////////
	//FAST-FREAK
	//////////////////////////
	GFTT_FREAK::GFTT_FREAK(const ParametersMap & parameters) :
	GFTT(parameters),
	orientationNormalized_(Parameters::defaultFREAKOrientationNormalized()),
	scaleNormalized_(Parameters::defaultFREAKScaleNormalized()),
	patternScale_(Parameters::defaultFREAKPatternScale()),
	nOctaves_(Parameters::defaultFREAKNOctaves()),
	_freak(0)
	{
		parseParameters(parameters);
	}

	GFTT_FREAK::~GFTT_FREAK()
	{
		if(_freak)
		{
			delete _freak;
		}
	}

	void GFTT_FREAK::parseParameters(const ParametersMap & parameters)
	{
		GFTT::parseParameters(parameters);

		Parameters::parse(parameters, Parameters::kFREAKOrientationNormalized(), orientationNormalized_);
		Parameters::parse(parameters, Parameters::kFREAKScaleNormalized(), scaleNormalized_);
		Parameters::parse(parameters, Parameters::kFREAKPatternScale(), patternScale_);
		Parameters::parse(parameters, Parameters::kFREAKNOctaves(), nOctaves_);

		if(_freak)
		{
			delete _freak;
			_freak = 0;
		}

		_freak = new cv::FREAK(orientationNormalized_, scaleNormalized_, patternScale_, nOctaves_);
	}

	cv::Mat GFTT_FREAK::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		_freak->compute(image, keypoints, descriptors);
		return descriptors;
	}

	//////////////////////////
	//BRISK
	//////////////////////////
	BRISK::BRISK(const ParametersMap & parameters) :
	thresh_(Parameters::defaultBRISKThreshold()),
	octaves_(Parameters::defaultBRISKNOctaves()),
	patternScale_(Parameters::defaultBRISKPatternScale()),
	brisk_(0)
	{
		parseParameters(parameters);
	}

	BRISK::~BRISK()
	{
		if(brisk_)
		{
			delete brisk_;
		}
	}

	void BRISK::parseParameters(const ParametersMap & parameters)
	{
		Parameters::parse(parameters, Parameters::kBRISKThreshold(), thresh_);
		Parameters::parse(parameters, Parameters::kBRISKNOctaves(), octaves_);
		Parameters::parse(parameters, Parameters::kBRISKPatternScale(), patternScale_);

		if(brisk_)
		{
			delete brisk_;
			brisk_ = 0;
		}

		brisk_ = new cv::BRISK(thresh_, octaves_, patternScale_);
	}

	std::vector<cv::KeyPoint> BRISK::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat imgRoi(image, roi);
		brisk_->detect(imgRoi, keypoints); // Opencv keypoints
		return keypoints;
	}

	cv::Mat BRISK::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
	{
		UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
		cv::Mat descriptors;
		brisk_->compute(image, keypoints, descriptors);
		return descriptors;
	}

	FixedPartition::FixedPartition( const long nFeatures, const double radius, const bool overlapse )
	{
		this->nFeatures = nFeatures;
		this->radius = radius;
		this->overlapse = overlapse;
	}

	void FixedPartition::detect( const cv::Mat image, std::vector<cv::KeyPoint> &keypoints )
	{
		double volume = double( image.cols * image.rows ) / double( this->nFeatures );
		double radius = sqrt( volume / M_PI );

		if( this->radius < 1 ) this->radius = radius;

		double step = this->radius;

		if( !this->overlapse ) step *= 2.0;

		for( double y = this->radius; y < image.rows; y += step )
		{
			for( double x = this->radius; x < image.cols; x += step )
			{
				cv::KeyPoint kp;
				kp.pt.x = x;
				kp.pt.y = y;
				kp.size = this->radius * 2.0;
				keypoints.push_back( kp );
			}
		}
	}

	DekWan::DekWan( const ParametersMap &parameters )
	{
		this->parameters = parameters;
		parseParameters( parameters );
	}

	DekWan::~DekWan()
	{

	}

	std::vector<cv::KeyPoint> DekWan::generateKeypointsImpl( const cv::Mat &image, const cv::Rect &roi ) const
	{
		std::vector<cv::KeyPoint> keypoints;
		std::vector<std::vector<cv::Point>> keypoints_sets;
		cv::Mat outImage = image.clone();

		switch( titikUtama )
		{
			case Parameters::SURF:
			surf->operator()( image, cv::Mat(), keypoints );
			break;

			case Parameters::SIFT:
			sift->operator()( image, cv::Mat(), keypoints );
			break;

			case Parameters::FAST:
			fast->detect( image, keypoints );
			break;

			case Parameters::FASTX:
			{
				// FASTX
				int threshold = Parameters::defaultFASTXThreshold();
				bool nonmaxSuppression = Parameters::defaultFASTXNonMaxSuppression();
				int type = Parameters::defaultFASTXType();
				Parameters::parse( parameters, Parameters::kFASTXThreshold(), threshold );
				Parameters::parse( parameters, Parameters::kFASTXNonMaxSuppression(), nonmaxSuppression );
				Parameters::parse( parameters, Parameters::kFASTXType(), type );
				cv::FASTX( image, keypoints, threshold, nonmaxSuppression, type );
				break;
			}

			case Parameters::MSER:
			{
				mser->operator()( image, keypoints_sets, cv::Mat() );

				double sum_size = 0;

				for( auto i : keypoints_sets )
				{
					sum_size += i.size();
				}

				for( unsigned long i = 0; i < keypoints_sets.size(); i++ )
				{
					cv::KeyPoint keypoint;
					keypoint.pt.x = 0;
					keypoint.pt.y = 0;

					for( auto j : keypoints_sets[i] )
					{
						keypoint.pt.x += j.x;
						keypoint.pt.y += j.y;
					}

					keypoint.pt.x /= keypoints_sets[i].size();
					keypoint.pt.y /= keypoints_sets[i].size();
					keypoint.size = 0;

					for( auto j : keypoints_sets[i] )
					{
						keypoint.size += std::sqrt( std::pow( keypoint.pt.x - j.x, 2 ) + std::pow( keypoint.pt.y - j.y, 2 ) );
						keypoint.angle += atan2( keypoint.pt.y - j.y, keypoint.pt.x - j.x ) * 180 / M_PI;
					}

					keypoint.size /= keypoints_sets[i].size();
					keypoint.angle /= keypoints_sets[i].size();

					keypoint.size *= 2;
					keypoint.response = keypoints_sets[i].size() / sum_size;
					keypoint.class_id = i;

					keypoints.push_back( keypoint );
				}
				break;
			}

			case Parameters::ORB:
			orb->operator()( image, cv::Mat(), keypoints );
			break;

			case Parameters::BRISK:
			brisk->operator()( image, cv::Mat(), keypoints );
			break;

			case Parameters::STAR:
			star->detect( image, keypoints );
			break;

			case Parameters::GFTT:
			gftt->detect( image, keypoints );
			break;

			case Parameters::DENSE:
			{
				dense->detect( image, keypoints );
				double diameter = std::sqrt( ( image.cols * image.rows / keypoints.size() ) / M_PI ) * 2.0;

				for( auto& i : keypoints )
				{
					i.size = diameter;
				}
				break;
			}

			case Parameters::SIMPLEBLOB:
			sbd->detect( image, keypoints );
			break;

			case Parameters::FIXED_PARTITION:
			fpartition->detect( image, keypoints );
			break;

			case Parameters::GAFD:
			gafd->detect( image, keypoints );
			break;

			case Parameters::PAFD:
			pafd->detect( image, keypoints );
			break;

			default:
			#ifdef WITH_NONFREE
			surf->operator()( image, cv::Mat(), keypoints );
			#else
			orb->operator()( image, cv::Mat(), keypoints );
			#endif
			break;
		}

		if( keypoints.size() < 1 )
		{
			cv::KeyPoint keypoint;
			keypoint.pt.x = image.cols / 2.0;
			keypoint.pt.y = image.rows / 2.0;
			keypoint.size = sqrt( ( image.cols * image.rows ) / M_PI ) * 2;
			keypoints.push_back( keypoint );
		}

		// cv::drawKeypoints( outImage, keypoints, outImage, cv::Scalar::all( -1 ), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		// cv::imshow( "outImage", outImage );
		// cv::waitKey( true );
		// cv::imwrite( "outImage.jpg", outImage );

		return keypoints;
	}

	cv::Mat DekWan::generateDescriptorsImpl( const cv::Mat &image, std::vector<cv::KeyPoint> & keypoints ) const
	{
		cv::Mat descriptors;

		switch( diskriptor )
		{
			case Parameters::SURF:
			surf->operator()( image, cv::Mat(), keypoints, descriptors, true );
			break;

			case Parameters::SIFT:
			sift->operator()( image, cv::Mat(), keypoints, descriptors, true );
			break;

			case Parameters::ORB:
			orb->operator()( image, cv::Mat(), keypoints, descriptors, true );
			break;

			case Parameters::BRISK:
			brisk->operator()( image, cv::Mat(), keypoints, descriptors, true );
			break;

			case Parameters::FREAK:
			freak->compute( image, keypoints, descriptors );
			break;

			case Parameters::SIFTDESC:
			siftdesc->compute( image, keypoints, descriptors );
			break;

			case Parameters::OCDE:
			ocde->compute( image, keypoints, descriptors );
			break;

			case Parameters::BRIEF:
			brief->compute( image, keypoints, descriptors );
			break;

			case Parameters::CSD:
			csd->compute( image, keypoints, descriptors );
			break;

			case Parameters::SCD:
			scd->compute( image, keypoints, descriptors );
			break;

			case Parameters::GOFGOP:
			gofgop->compute( image, keypoints, descriptors );
			break;

			case Parameters::DCD:
			dcd->compute( image, keypoints, descriptors );
			break;

			case Parameters::CLD:
			cld->compute( image, keypoints, descriptors );
			break;

			case Parameters::EHD:
			ehd->compute( image, keypoints, descriptors );
			break;

			case Parameters::HTD:
			htd->compute( image, keypoints, descriptors );
			break;

			case Parameters::CSHD:
			cshd->compute( image, keypoints, descriptors );
			break;

			case Parameters::RSD:
			rsd->compute( image, keypoints, descriptors );
			break;

			case Parameters::FRD:
			frd->compute( image, keypoints, descriptors );
			break;

			default:
			#ifdef WITH_NONFREE
			surf->operator()( image, cv::Mat(), keypoints, descriptors, true );
			#else
			orb->operator()( image, cv::Mat(), keypoints, descriptors, true );
			#endif
			break;
		}

		// std::cout << descriptors << std::endl;
		// cv::imshow( "descriptors", descriptors );
		// cv::waitKey( false );
		// exit( true );

		return descriptors;
	}

	void DekWan::parseParameters( const ParametersMap &parameters )
	{
		// WRT-Map
		Parameters::parse( parameters, Parameters::kDekWanKeyPoint(), titikUtama );
		Parameters::parse( parameters, Parameters::kDekWanDescriptor(), diskriptor );

		// SURF
		double hessianThreshold = Parameters::defaultSURFHessianThreshold();
		int nOctaves = Parameters::defaultSURFNOctaves();
		int nOctaveLayers = Parameters::defaultSURFNOctaveLayers();
		bool extended = Parameters::defaultSURFExtended();
		bool upright = Parameters::defaultSURFUpright();
		Parameters::parse( parameters, Parameters::kSURFHessianThreshold(), hessianThreshold );
		Parameters::parse( parameters, Parameters::kSURFNOctaves(), nOctaves );
		Parameters::parse( parameters, Parameters::kSURFNOctaveLayers(), nOctaveLayers );
		Parameters::parse( parameters, Parameters::kSURFExtended(), extended );
		Parameters::parse( parameters, Parameters::kSURFUpright(), upright );
		surf.reset( new cv::SURF( hessianThreshold, nOctaves, nOctaveLayers, extended, upright ) );

		// SIFT
		int nFeatures = Parameters::defaultSIFTNFeatures();
		nOctaveLayers = Parameters::defaultSIFTNOctaveLayers();
		double contrastThreshold = Parameters::defaultSIFTContrastThreshold();
		double edgeThreshold = Parameters::defaultSIFTEdgeThreshold();
		double sigma = Parameters::defaultSIFTSigma();
		Parameters::parse( parameters, Parameters::kSIFTNFeatures(), nFeatures );
		Parameters::parse( parameters, Parameters::kSIFTNOctaveLayers(), nOctaveLayers );
		Parameters::parse( parameters, Parameters::kSIFTContrastThreshold(), contrastThreshold );
		Parameters::parse( parameters, Parameters::kSIFTEdgeThreshold(), edgeThreshold );
		Parameters::parse( parameters, Parameters::kSIFTSigma(), sigma );
		sift.reset( new cv::SIFT( nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma ) );

		// ORB
		nFeatures = Parameters::defaultORBNFeatures();
		double scaleFactor = Parameters::defaultORBScaleFactor();
		int nlevels = Parameters::defaultORBNLevels();
		edgeThreshold = Parameters::defaultORBEdgeThreshold();
		int firstLevel = Parameters::defaultORBFirstLevel();
		int wta_k = Parameters::defaultORBWTA_K();
		int scoreType = Parameters::defaultORBScoreType();
		int patchSize = Parameters::defaultORBPatchSize();
		Parameters::parse( parameters, Parameters::kORBNFeatures(), nFeatures );
		Parameters::parse( parameters, Parameters::kORBScaleFactor(), scaleFactor );
		Parameters::parse( parameters, Parameters::kORBNLevels(), nlevels );
		Parameters::parse( parameters, Parameters::kORBEdgeThreshold(), edgeThreshold );
		Parameters::parse( parameters, Parameters::kORBFirstLevel(), firstLevel );
		Parameters::parse( parameters, Parameters::kORBWTA_K(), wta_k );
		Parameters::parse( parameters, Parameters::kORBScoreType(), scoreType );
		Parameters::parse( parameters, Parameters::kORBPatchSize(), patchSize );
		orb.reset( new cv::ORB( nFeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize ) );

		// FAST
		int threshold = Parameters::defaultFASTThreshold();
		bool nonmaxSuppression = Parameters::defaultFASTNonMaxSuppression();
		Parameters::parse( parameters, Parameters::kFASTThreshold(), threshold );
		Parameters::parse( parameters, Parameters::kFASTNonMaxSuppression(), nonmaxSuppression );
		fast.reset( new cv::FastFeatureDetector( threshold, nonmaxSuppression ) );

		// FREAK
		bool orientationNormalized = Parameters::defaultFREAKOrientationNormalized();
		bool scaleNormalized = Parameters::defaultFREAKScaleNormalized();
		double patternScale = Parameters::defaultFREAKPatternScale();
		nOctaves = Parameters::defaultFREAKNOctaves();
		Parameters::parse( parameters, Parameters::kFREAKOrientationNormalized(), orientationNormalized );
		Parameters::parse( parameters, Parameters::kFREAKScaleNormalized(), scaleNormalized );
		Parameters::parse( parameters, Parameters::kFREAKPatternScale(), patternScale );
		Parameters::parse( parameters, Parameters::kFREAKNOctaves(), nOctaves );
		freak.reset( new cv::FREAK( orientationNormalized, scaleNormalized, patternScale, nOctaves ) );

		// BRIEF
		int bytes = Parameters::defaultBRIEFBytes();
		Parameters::parse( parameters, Parameters::kBRIEFBytes(), bytes );
		brief.reset( new cv::BriefDescriptorExtractor( bytes ) );

		// GFTT
		int maxCorners = Parameters::defaultGFTTMaxCorners();
		double qualityLevel = Parameters::defaultGFTTQualityLevel();
		double minDistance = Parameters::defaultGFTTMinDistance();
		int blockSize = Parameters::defaultGFTTBlockSize();
		bool useHarrisDetector = Parameters::defaultGFTTUseHarrisDetector();
		double k = Parameters::defaultGFTTK();
		Parameters::parse( parameters, Parameters::kGFTTMaxCorners(), maxCorners );
		Parameters::parse( parameters, Parameters::kGFTTQualityLevel(), qualityLevel );
		Parameters::parse( parameters, Parameters::kGFTTMinDistance(), minDistance );
		Parameters::parse( parameters, Parameters::kGFTTBlockSize(), blockSize );
		Parameters::parse( parameters, Parameters::kGFTTUseHarrisDetector(), useHarrisDetector );
		Parameters::parse( parameters, Parameters::kGFTTK(), k );
		gftt.reset( new cv::GoodFeaturesToTrackDetector( maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k ) );

		// MSER
		int delta = Parameters::defaultMSERDelta();
		int minArea = Parameters::defaultMSERMinArea();
		int maxArea = Parameters::defaultMSERMaxArea();
		double maxVariation = Parameters::defaultMSERMaxVariation();
		double minDiversity = Parameters::defaultMSERMinDiversity();
		int maxEvolution = Parameters::defaultMSERMaxEvolution();
		double areaThreshold = Parameters::defaultMSERAreaThreshold();
		double minMargin = Parameters::defaultMSERMinMargin();
		int edgeBlurSize = Parameters::defaultMSEREdgeBlurSize();
		Parameters::parse( parameters, Parameters::kMSERDelta(), delta );
		Parameters::parse( parameters, Parameters::kMSERMinArea(), minArea );
		Parameters::parse( parameters, Parameters::kMSERMaxArea(), maxArea );
		Parameters::parse( parameters, Parameters::kMSERMaxVariation(), maxVariation );
		Parameters::parse( parameters, Parameters::kMSERMinDiversity(), minDiversity );
		Parameters::parse( parameters, Parameters::kMSERMaxEvolution(), maxEvolution );
		Parameters::parse( parameters, Parameters::kMSERAreaThreshold(), areaThreshold );
		Parameters::parse( parameters, Parameters::kMSERMinMargin(), minMargin );
		Parameters::parse( parameters, Parameters::kMSEREdgeBlurSize(), edgeBlurSize );
		mser.reset( new cv::MSER( delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution, areaThreshold, minMargin, edgeBlurSize ) );

		// BRISK
		threshold = Parameters::defaultBRISKThreshold();
		nOctaves = Parameters::defaultBRISKNOctaves();
		patternScale = Parameters::defaultBRISKPatternScale();
		Parameters::parse( parameters, Parameters::kBRISKThreshold(), threshold );
		Parameters::parse( parameters, Parameters::kBRISKNOctaves(), nOctaves );
		Parameters::parse( parameters, Parameters::kBRISKPatternScale(), patternScale );
		brisk.reset( new cv::BRISK( threshold, nOctaves, patternScale ) );

		// STAR
		int maxSize = Parameters::defaultSTARMaxSize();
		int responseThreshold = Parameters::defaultSTARResponseThreshold();
		int lineThresholdProjected = Parameters::defaultSTARLineThresholdProjected();
		int lineThresholdBinarized = Parameters::defaultSTARLineThresholdBinarized();
		int suppressNonmaxSize = Parameters::defaultSTARSuppressNonmaxSize();
		Parameters::parse( parameters, Parameters::kSTARMaxSize(), maxSize );
		Parameters::parse( parameters, Parameters::kSTARResponseThreshold(), responseThreshold );
		Parameters::parse( parameters, Parameters::kSTARLineThresholdProjected(), lineThresholdProjected );
		Parameters::parse( parameters, Parameters::kSTARLineThresholdBinarized(), lineThresholdBinarized );
		Parameters::parse( parameters, Parameters::kSTARSuppressNonmaxSize(), suppressNonmaxSize );
		star.reset( new cv::StarFeatureDetector( maxSize, responseThreshold, lineThresholdProjected, lineThresholdBinarized, suppressNonmaxSize ) );

		// DENSE
		double initFeatureScale = Parameters::defaultDenseInitFeatureScale();
		int featureScaleLevels = Parameters::defaultDenseFeatureScaleLevels();
		double featureScaleMul = Parameters::defaultDenseFeatureScaleMul();
		int initXyStep = Parameters::defaultDenseInitXyStep();
		int initImgBound = Parameters::defaultDenseInitImgBound();
		bool varyXyStepWithScale = Parameters::defaultDenseVaryXyStepWithScale();
		bool varyImgBoundWithScale = Parameters::defaultDenseVaryImgBoundWithScale();
		Parameters::parse( parameters, Parameters::kDenseInitFeatureScale(), initFeatureScale );
		Parameters::parse( parameters, Parameters::kDenseFeatureScaleLevels(), featureScaleLevels );
		Parameters::parse( parameters, Parameters::kDenseFeatureScaleMul(), featureScaleMul );
		Parameters::parse( parameters, Parameters::kDenseInitXyStep(), initXyStep );
		Parameters::parse( parameters, Parameters::kDenseInitImgBound(), initImgBound );
		Parameters::parse( parameters, Parameters::kDenseVaryXyStepWithScale(), varyXyStepWithScale );
		Parameters::parse( parameters, Parameters::kDenseVaryImgBoundWithScale(), varyImgBoundWithScale );
		dense.reset( new cv::DenseFeatureDetector( initFeatureScale, featureScaleLevels, featureScaleMul, initXyStep, initImgBound, varyXyStepWithScale, varyImgBoundWithScale ) );

		// Simple Blob Detector
		cv::SimpleBlobDetector::Params sbdp;
		sbdp.thresholdStep = Parameters::defaultSBDThresholdStep();
		sbdp.minThreshold = Parameters::defaultSBDMinThreshold();
		sbdp.maxThreshold = Parameters::defaultSBDMaxThreshold();
		sbdp.minRepeatability = Parameters::defaultSBDMinRepeatability();
		sbdp.minDistBetweenBlobs = Parameters::defaultSBDMinDistBetweenBlobs();
		sbdp.filterByColor = Parameters::defaultSBDFilterByColor();
		sbdp.blobColor = Parameters::defaultSBDBlobColor();
		sbdp.filterByArea = Parameters::defaultSBDFilterByArea();
		sbdp.minArea = Parameters::defaultSBDMinArea();
		sbdp.maxArea = Parameters::defaultSBDMaxArea();
		sbdp.filterByCircularity = Parameters::defaultSBDFilterByCircularity();
		sbdp.minCircularity = Parameters::defaultSBDMinCircularity();
		sbdp.maxCircularity = Parameters::defaultSBDMaxCircularity();
		sbdp.filterByInertia = Parameters::defaultSBDFilterByInertia();
		sbdp.minInertiaRatio = Parameters::defaultSBDMinInertiaRatio();
		sbdp.maxInertiaRatio = Parameters::defaultSBDMaxInertiaRatio();
		sbdp.filterByConvexity = Parameters::defaultSBDFilterByConvexity();
		sbdp.minConvexity = Parameters::defaultSBDMinConvexity();
		sbdp.maxConvexity = Parameters::defaultSBDMaxConvexity();
		int i = 0;
		Parameters::parse( parameters, Parameters::kSBDThresholdStep(), sbdp.thresholdStep );
		Parameters::parse( parameters, Parameters::kSBDMinThreshold(), sbdp.minThreshold );
		Parameters::parse( parameters, Parameters::kSBDMaxThreshold(), sbdp.maxThreshold );
		Parameters::parse( parameters, Parameters::kSBDMinRepeatability(), i );
		sbdp.minRepeatability = i;
		Parameters::parse( parameters, Parameters::kSBDMinDistBetweenBlobs(), sbdp.minDistBetweenBlobs );
		Parameters::parse( parameters, Parameters::kSBDFilterByColor(), sbdp.filterByColor );
		Parameters::parse( parameters, Parameters::kSBDBlobColor(), i );
		sbdp.blobColor = i;
		Parameters::parse( parameters, Parameters::kSBDFilterByArea(), sbdp.filterByArea );
		Parameters::parse( parameters, Parameters::kSBDMinArea(), sbdp.minArea );
		Parameters::parse( parameters, Parameters::kSBDMaxArea(), sbdp.maxArea );
		Parameters::parse( parameters, Parameters::kSBDFilterByCircularity(), sbdp.filterByCircularity );
		Parameters::parse( parameters, Parameters::kSBDMinCircularity(), sbdp.minCircularity );
		Parameters::parse( parameters, Parameters::kSBDMaxCircularity(), sbdp.maxCircularity );
		Parameters::parse( parameters, Parameters::kSBDFilterByInertia(), sbdp.filterByInertia );
		Parameters::parse( parameters, Parameters::kSBDMinInertiaRatio(), sbdp.minInertiaRatio );
		Parameters::parse( parameters, Parameters::kSBDMaxInertiaRatio(), sbdp.maxInertiaRatio );
		Parameters::parse( parameters, Parameters::kSBDFilterByConvexity(), sbdp.filterByConvexity );
		Parameters::parse( parameters, Parameters::kSBDMinConvexity(), sbdp.minConvexity );
		Parameters::parse( parameters, Parameters::kSBDMaxConvexity(), sbdp.maxConvexity );
		sbd.reset( new cv::SimpleBlobDetector( sbdp ) );

		// Fixed Partition
		nFeatures = Parameters::defaultFPartitionNFeatures();
		int radius = Parameters::defaultFPartitionRadius();
		bool overlapse = Parameters::defaultFPartitionOverlapse();
		Parameters::parse( parameters, Parameters::kFPartitionNFeatures(), nFeatures );
		Parameters::parse( parameters, Parameters::kFPartitionRadius(), radius );
		Parameters::parse( parameters, Parameters::kFPartitionOverlapse(), overlapse );
		fpartition.reset( new FixedPartition( nFeatures, radius, overlapse ) );

		// Sift Descriptor
		int dims = Parameters::defaultAngleSiftDims();
		int bins = Parameters::defaultAngleSiftBins();
		double orientation = Parameters::defaultAngleSiftOrientation();
		Parameters::parse( parameters, Parameters::kAngleSiftDims(), dims );
		Parameters::parse( parameters, Parameters::kAngleSiftBins(), bins );
		Parameters::parse( parameters, Parameters::kAngleSiftOrientation(), orientation );
		siftdesc.reset( new AngleSift( dims, bins, orientation ) );

		// Grid Adapted Feature Detector
		int detector = Parameters::defaultGAFDDetector();
		nFeatures = Parameters::defaultGAFDNFeatures();
		int gridRows = Parameters::defaultGAFDGridRows();
		int gridCols = Parameters::defaultGAFDGridCols();
		Parameters::parse( parameters, Parameters::kGAFDDetector(), detector );
		Parameters::parse( parameters, Parameters::kGAFDNFeatures(), nFeatures );
		Parameters::parse( parameters, Parameters::kGAFDGridRows(), gridRows );
		Parameters::parse( parameters, Parameters::kGAFDGridCols(), gridCols );
		cv::Ptr<cv::FeatureDetector> fDetector = setDetector( detector );
		gafd.reset( new cv::GridAdaptedFeatureDetector( fDetector, nFeatures, gridRows, gridCols ) );

		// Pyramid Adapted Feature Detector
		detector = Parameters::defaultPAFDDetector();
		nlevels = Parameters::defaultPAFDNLevels();
		Parameters::parse( parameters, Parameters::kPAFDDetector(), detector );
		Parameters::parse( parameters, Parameters::kPAFDNLevels(), nlevels );
		fDetector = setDetector( detector );
		pafd.reset( new cv::PyramidAdaptedFeatureDetector( fDetector, nlevels ) );

		// Opponent Color Descriptor Extractor
		int extractor = Parameters::defaultOCDEExtractor();
		Parameters::parse( parameters, Parameters::kOCDEExtractor(), extractor );
		cv::Ptr<cv::DescriptorExtractor> dExtractor = setExtractor( extractor );
		ocde.reset( new cv::OpponentColorDescriptorExtractor( dExtractor ) );

		// MPEG7
		// Color Structure Descriptor
		int descSize = Parameters::defaultCSDDescSize();
		Parameters::parse( parameters, Parameters::kCSDDescSize(), descSize );
		csd.reset( new dekwan::ColorStructureDescriptor( descSize ) );

		// Scalable Color Descriptor
		int numCoeff = Parameters::defaultSCDNumCoeff();
		int bitPlanesDiscarded = Parameters::defaultSCDBitPlanesDiscarded();
		Parameters::parse( parameters, Parameters::kSCDNumCoeff(), numCoeff );
		Parameters::parse( parameters, Parameters::kSCDBitPlanesDiscarded(), bitPlanesDiscarded );
		scd.reset( new dekwan::ScalableColorDescriptor( numCoeff, bitPlanesDiscarded ) );

		// GoFGoP Color Descriptor
		numCoeff = Parameters::defaultGoFGoPNumCoeff();
		bitPlanesDiscarded = Parameters::defaultGoFGoPBitPlanesDiscarded();
		Parameters::parse( parameters, Parameters::kGoFGoPNumCoeff(), numCoeff );
		Parameters::parse( parameters, Parameters::kGoFGoPBitPlanesDiscarded(), bitPlanesDiscarded );
		gofgop.reset( new dekwan::GoFGoPColorDescriptor( numCoeff, bitPlanesDiscarded ) );

		// Dominant Color Descriptor
		bool normalize = Parameters::defaultDCDNormalize();
		bool variance = Parameters::defaultDCDVariance();
		bool spatial = Parameters::defaultDCDSpatial();
		int bin1 = Parameters::defaultDCDBin1();
		int bin2 = Parameters::defaultDCDBin2();
		int bin3 = Parameters::defaultDCDBin3();
		Parameters::parse( parameters, Parameters::kDCDNormalize(), normalize );
		Parameters::parse( parameters, Parameters::kDCDVariance(), variance );
		Parameters::parse( parameters, Parameters::kDCDSpatial(), spatial );
		Parameters::parse( parameters, Parameters::kDCDBin1(), bin1 );
		Parameters::parse( parameters, Parameters::kDCDBin2(), bin2 );
		Parameters::parse( parameters, Parameters::kDCDBin3(), bin3 );
		dcd.reset( new dekwan::DominantColorDescriptor( normalize, variance, spatial, bin1, bin2, bin3 ) );

		// Color Layout Descriptor
		int numberOfYCoeff = Parameters::defaultCLDNumberOfYCoeff();
		int numberOfCCoeff = Parameters::defaultCLDNumberOfCCoeff();
		Parameters::parse( parameters, Parameters::kCLDNumberOfYCoeff(), numberOfYCoeff );
		Parameters::parse( parameters, Parameters::kCLDNumberOfCCoeff(), numberOfCCoeff );
		cld.reset( new dekwan::ColorLayoutDescriptor( numberOfYCoeff, numberOfCCoeff ) );

		// Edge Histogram Descriptor
		ehd.reset( new dekwan::EdgeHistogramDescriptor() );

		// Homogeneous Texture Descriptor
		bool layerFlag = Parameters::defaultHTDLayerFlag();
		Parameters::parse( parameters, Parameters::kHTDLayerFlag(), layerFlag );
		htd.reset( new dekwan::HomogeneousTextureDescriptor( layerFlag ) );

		// Contour Shape Descriptor
		double ratio = Parameters::defaultCSDRatio();
		threshold = Parameters::defaultCSDThreshold();
		int apertureSize = Parameters::defaultCSDApertureSize();
		int kernel = Parameters::defaultCSDKernel();
		Parameters::parse( parameters, Parameters::kCSDRatio(), ratio );
		Parameters::parse( parameters, Parameters::kCSDThreshold(), threshold );
		Parameters::parse( parameters, Parameters::kCSDApertureSize(), apertureSize );
		Parameters::parse( parameters, Parameters::kCSDKernel(), kernel );
		cshd.reset( new dekwan::ContourShapeDescriptor( ratio, threshold, apertureSize, kernel ) );

		// Region Shape Descriptor
		ratio = Parameters::defaultRSDRatio();
		threshold = Parameters::defaultRSDThreshold();
		apertureSize = Parameters::defaultRSDApertureSize();
		kernel = Parameters::defaultRSDKernel();
		Parameters::parse( parameters, Parameters::kRSDRatio(), ratio );
		Parameters::parse( parameters, Parameters::kRSDThreshold(), threshold );
		Parameters::parse( parameters, Parameters::kRSDApertureSize(), apertureSize );
		Parameters::parse( parameters, Parameters::kRSDKernel(), kernel );
		rsd.reset( new dekwan::RegionShapeDescriptor( ratio, threshold, apertureSize, kernel ) );

		// Face Recognition Descriptor
		frd.reset( new dekwan::FaceRecognitionDescriptor() );
	}

	cv::Ptr<cv::FeatureDetector> DekWan::setDetector( const int detector )
	{
		switch( detector )
		{
			case Parameters::SURF:
			return surf.get();
			break;

			case Parameters::SIFT:
			return sift.get();
			break;

			case Parameters::FAST:
			return fast.get();
			break;

			case Parameters::MSER:
			return mser.get();
			break;

			case Parameters::ORB:
			return orb.get();
			break;

			case Parameters::BRISK:
			return brisk.get();
			break;

			case Parameters::STAR:
			return star.get();
			break;

			case Parameters::GFTT:
			return gftt.get();
			break;

			case Parameters::DENSE:
			return dense.get();
			break;

			case Parameters::SIMPLEBLOB:
			return sbd.get();
			break;

			default:
			#ifdef WITH_NONFREE
			return surf.get();
			#else
			return orb.get();
			#endif
			break;
		}
	}

	cv::Ptr<cv::DescriptorExtractor> DekWan::setExtractor( const int extractor )
	{
		switch( extractor )
		{
			case Parameters::SURF:
			return surf.get();
			break;

			case Parameters::SIFT:
			return sift.get();
			break;

			case Parameters::ORB:
			return orb.get();
			break;

			case Parameters::BRISK:
			return brisk.get();
			break;

			case Parameters::FREAK:
			return freak.get();
			break;

			default:
			#ifdef WITH_NONFREE
			return surf.get();
			#else
			return orb.get();
			#endif
			break;
		}
	}
}
