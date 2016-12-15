/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <opencv.hpp>

namespace cv{

	struct CV_EXPORTS_W _HOGDescriptor;
	struct _HOGCache
	{
		struct BlockData
		{
			BlockData() : histOfs(0), imgOffset() {}
			int histOfs;
			Point imgOffset;
		};

		struct PixData
		{
			size_t gradOfs, qangleOfs;
			int histOfs[4];
			float histWeights[4];
			float gradWeight;
			int xind, yind;
		};

		_HOGCache();
		_HOGCache(const _HOGDescriptor* descriptor,
		    Size paddingTL, Size paddingBR,
			bool useCache, Size cacheStride);
		virtual ~_HOGCache() {};
		virtual void init(const _HOGDescriptor* descriptor,
			Size paddingTL, Size paddingBR,
			bool useCache, Size cacheStride);

		Size windowsInImage(Size imageSize, Size winStride) const;
		Rect getWindow(Size imageSize, Size winStride, int idx) const;

		const float* getBlock(Point pt, float* buf);
		virtual void normalizeBlockHistogram(float* histogram) const;

		vector<PixData> pixData;
		vector<BlockData> blockData;

		bool useCache;
		vector<int> ymaxCached;
		Size winSize, cacheStride;
		Size nblocks, ncells;
		int blockHistogramSize;
		int count1, count2, count4;
		Point imgoffset;
		Mat_<float> blockCache;
		Mat_<uchar> blockCacheFlags;

		const _HOGDescriptor* descriptor;
	};

	struct CV_EXPORTS_W _HOGDescriptor
	{
	public:
		enum { L2Hys = 0 };
		enum { DEFAULT_NLEVELS = 64 };

		CV_WRAP _HOGDescriptor() : winSize(64, 128), blockSize(16, 16), blockStride(8, 8),
			cellSize(8, 8), nbins(9), derivAperture(1), winSigma(-1),
			histogramNormType(_HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
			nlevels(_HOGDescriptor::DEFAULT_NLEVELS)
		{}

		CV_WRAP _HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
			Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1,
			int _histogramNormType = HOGDescriptor::L2Hys,
			double _L2HysThreshold = 0.2, bool _gammaCorrection = false,
			int _nlevels = HOGDescriptor::DEFAULT_NLEVELS);

		CV_WRAP _HOGDescriptor(const String& filename)
		{
			load(filename);
		}

		_HOGDescriptor(const _HOGDescriptor& d)
		{
			d.copyTo(*this);
		}

		virtual ~_HOGDescriptor() { if (mapbuf) delete[] mapbuf; }

		CV_WRAP size_t getDescriptorSize() const;
		CV_WRAP bool checkDetectorSize() const;
		CV_WRAP double getWinSigma() const;

		CV_WRAP virtual void setSVMDetector(InputArray _svmdetector);

		virtual bool read(FileNode& fn);
		virtual void write(FileStorage& fs, const String& objname) const;

		CV_WRAP virtual bool load(const String& filename, const String& objname = String());
		CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;
		virtual void copyTo(_HOGDescriptor& c) const;

		CV_WRAP virtual void compute(const Mat& img,
			CV_OUT vector<float>& descriptors,
			Size winStride = Size(), Size padding = Size(),
			const vector<Point>& locations = vector<Point>()) const;
		//with found weights output
		CV_WRAP virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
			CV_OUT vector<double>& weights,
			double hitThreshold = 0, Size winStride = Size(),
			Size padding = Size(),
			const vector<Point>& searchLocations = vector<Point>()) const;
		//without found weights output
		virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
			double hitThreshold = 0, Size winStride = Size(),
			Size padding = Size(),
			const vector<Point>& searchLocations = vector<Point>()) const;
		//with result weights output
		CV_WRAP virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
			CV_OUT vector<double>& foundWeights, double hitThreshold = 0,
			Size winStride = Size(), Size padding = Size(), double scale = 1.05,
			double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;
		//without found weights output
		virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
			double hitThreshold = 0, Size winStride = Size(),
			Size padding = Size(), double scale = 1.05,
			double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

		CV_WRAP void setXYMap(const int width, const int height, const Size& paddingTL, const Size& paddingBR);
		CV_WRAP virtual void computeGradient(const Mat& img) const;
		CV_WRAP virtual void computegradandangle(const float* dx, const float* dy, CV_OUT float* grad,
			CV_OUT uchar* angle, int numcol) const;

		CV_WRAP static vector<float> getDefaultPeopleDetector();
		CV_WRAP static vector<float> getDaimlerPeopleDetector();

		CV_PROP Size winSize;
		CV_PROP Size blockSize;
		CV_PROP Size blockStride;
		CV_PROP Size cellSize;
		CV_PROP int nbins;
		CV_PROP int derivAperture;
		CV_PROP double winSigma;
		CV_PROP int histogramNormType;
		CV_PROP double L2HysThreshold;
		CV_PROP bool gammaCorrection;
		CV_PROP vector<float> svmDetector;
		CV_PROP int nlevels;
		CV_PROP float* orientationX;
		CV_PROP float* orientationY;
		CV_PROP Mat_<float> _lut;
		CV_PROP _HOGCache cache;
		CV_PROP int* xmap;
		CV_PROP int* ymap;
		CV_PROP Mat grad, qangle;
		CV_PROP int* mapbuf;
		CV_PROP float anglescale;
		CV_PROP int rawBlockSize;

		// evaluate specified ROI and return confidence value for each location
		void detectROI(const cv::Mat& img, const vector<cv::Point> &locations,
			CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
			double hitThreshold = 0, cv::Size winStride = Size(),
			cv::Size padding = Size()) const;

		// evaluate specified ROI and return confidence value for each location in multiple scales
		void detectMultiScaleROI(const cv::Mat& img,
			CV_OUT std::vector<cv::Rect>& foundLocations,
			std::vector<DetectionROI>& locations,
			double hitThreshold = 0,
			int groupThreshold = 0) const;

		// read/parse Dalal's alt model file
		void readALTModel(std::string modelfile);
		void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	};

}


