#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include<vector>
#include<functional>

namespace Index {
	namespace Util {
		// non-parallel foreach
		template<typename T>
		void ForEachPixel(cv::Mat& Mat, std::function<void(T&, const int)> Lambda) {
			int N = Mat.rows * Mat.cols;
			for (int i = 0; i < N; i++) {
				Lambda(Mat.at<T>(i), i);
			}
		}
	}

	// variation of information
	// Result: uchar
	// GroundTruth: uchar, GourndTruth after processed
	static double VI(cv::Mat& Result, cv::Mat& GroundTruth) {
		assert(Result.size() == GroundTruth.size());
		int K = 0, Total = GroundTruth.rows * GroundTruth.cols;
		std::vector<int>KMap(0x100, -1);
		Util::ForEachPixel<uchar>(GroundTruth,//GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			if (KMap[Pixel] < 0)KMap[Pixel] = K++;
		}
		);

		static auto H = [](const std::vector<int>& N, int Total)->double{
			double Ret = 0.0;
			for (int i = 0; i < N.size(); i++) {
				assert(N[i] > 0);
				Ret += ( double(N[i]) / Total ) * std::log((double(N[i]) / Total));
			}
			return -Ret;
		};

		static auto I = [](const std::vector<std::vector<int>>& N, const std::vector<int>& NR, const std::vector<int>& NG, int Total)->double {
			double Ret = 0.0;
			for (int i = 0; i < N.size(); i++) {
				double Sum = 0.0;
				for (int j = 0; j < N.size(); j++) {
					if (N[i][j] == 0)continue;
					assert(NR[i] > 0 && NG[j] > 0);
					//Sum += (double(N[i][j]) / Total)
					//	* std::log(
					//		(double(N[i][j]) / Total)
					//		* (double(NR[i]) / Total)
					//		* (double(NG[j]) / Total)
					//	);
					
					//Sum += (double(N[i][j]) / Total)
					//	* std::log(	(double(N[i][j]) / Total) )
					//	* (double(NR[i]) / Total)
					//	* (double(NG[j]) / Total);

					Sum += (double(N[i][j]) / Total)
						* std::log((double(N[i][j]) / Total) / ((double(NR[i]) / Total) * (double(NG[j]) / Total)));
				}
				Ret += Sum;
			}
			return Ret;
		};

		std::vector<int>NR(K, 0), NG(K, 0);
		std::vector<std::vector<int>> N(K, std::vector<int>(K, 0));
		Util::ForEachPixel<uchar>(Result, //Result.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			NR[KMap[Pixel]]++;
		}
		);
		Util::ForEachPixel<uchar>(GroundTruth, //GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			NG[KMap[Pixel]]++;
			auto RPixel = Result.at<uchar>(Position);
			auto KR = KMap[RPixel], KG = KMap[Pixel];
			N[KR][KG]++;
		}
		);

		//double Res = 0;
		//for (int i = 0; i < K; i++) {
		//	for (int j = 0; j < K; j++) {
		//		if (N[i][j] == 0)continue;
		//		double RIJ = double(N[i][j]) / Total;
		//		double Pi = double(NR[i]) / Total;
		//		double Qj = double(NG[j]) / Total;
		//		Res += RIJ * (std::log(RIJ / Pi) + std::log(RIJ / Qj));
		//	}
		//}
		//Res = -Res;
		//return Res;

		auto HR = H(NR, Total);
		auto HG = H(NG, Total); 
		auto I2 = 2 * I(N, NR, NG, Total);
		return HR + HG - I2;
	}
	
	// global consistency error
	// Result: uchar
	// GroundTruth: uchar, GourndTruth after processed
	static double GCE(cv::Mat& Result, cv::Mat& GroundTruth) {
		assert(Result.size() == GroundTruth.size());
		int K = 0, Total = GroundTruth.rows * GroundTruth.cols;
		std::vector<int>KMap(0x100, -1);
		Util::ForEachPixel<uchar>(GroundTruth, //GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			if (KMap[Pixel] < 0)KMap[Pixel] = K++;
		}
		);

		// NR[K]: number of pixels of class K in Result
		// NG[K]: number of pixels of class K in GroundTruth
		std::vector<int>NR(K, 0), NG(K, 0);
		// N[K1][K2]: number of pixels in intersection of K between Result and GroundTruth 
		std::vector<std::vector<int>> N(K, std::vector<int>(K, 0));
		// RR[K1][K2]: resigion size of (Result) - (GroundTruth) 
		// RG[K1][K2]: resigion size of (GroundTruth) - (Result) 
		std::vector<std::vector<int>> RR(K, std::vector<int>(K, 0)), RG(K, std::vector<int>(K, 0));
		Util::ForEachPixel<uchar>(Result, //Result.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			NR[KMap[Pixel]]++;
		}
		);
		Util::ForEachPixel<uchar>(GroundTruth, //GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			NG[KMap[Pixel]]++;
			auto RPixel = Result.at<uchar>(Position);
			N[KMap[RPixel]][KMap[Pixel]]++;
			if (KMap[RPixel] != KMap[Pixel])
				N[KMap[Pixel]][KMap[RPixel]]++;
		}
		);

		for (int i = 0; i < K; i++) {
			for (int j = 0; j < K; j++) {
				RR[i][j] = NR[i] - N[i][j];
				RG[i][j] = NG[i] - N[i][j];
			}
		}
		
		double SumLeft = 0.0, SumRight = 0.0;
		Util::ForEachPixel<uchar>(GroundTruth, //GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			auto RPixel = Result.at<uchar>(Position);
			auto RK = KMap[RPixel], GK = KMap[Pixel];
			SumLeft += double(RR[RK][GK]) / NR[RK];
			SumRight += double(RG[GK][RK]) / NG[GK];
		});

		return (1.0 / Total) * std::min(SumLeft, SumRight);
	}

	// Rand
	// Result: uchar
	// GroundTruth: uchar, GourndTruth after processed
	static double Rand(cv::Mat& Result, cv::Mat& GroundTruth) {
		assert(Result.size() == GroundTruth.size());
		int K = 0;
		std::vector<int>KMap(0x100, -1);
		Util::ForEachPixel<uchar>(GroundTruth, //GroundTruth.forEach<uchar>(
			[&](uchar& Pixel, const int Position) {
			if (KMap[Pixel] < 0)KMap[Pixel] = K++;
		}
		);

		//N[k1][k2] be the number of points having label k1 in Result and label k2 in GroundTruth.
		std::vector<std::vector<int>>N(K, std::vector<int>(K, 0));
		std::vector<int> NR(K, 0), NG(K, 0);
		Util::ForEachPixel<uchar>(Result,
			[&](uchar& Pixel, const int Position) {
			auto GPixel = GroundTruth.at<uchar>(Position);
			auto KR = KMap[Pixel];
			auto KG = KMap[GPixel];
			N[KR][KG]++;
			NR[KR]++;
			NG[KG]++;
		});

		double SumNu = 0.0, SumNv = 0.0, SumNuv = 0.0;
		for (int i = 0; i < K; i++) {
			SumNu += NR[i] * NR[i];
			SumNv += NG[i] * NG[i];
			for (int j = 0; j < K; j++) {
				SumNuv += N[i][j] * N[i][j];
			}
		}

		double Total = GroundTruth.rows * double(GroundTruth.cols);
		return 1 - (
			(0.5 * (SumNu + SumNv) - SumNuv) /
			(Total * (Total-1) * 0.5)
			);
	}

	// normalized probabilistic rand
	// Result: uchar
	// GroundTruth: uchar, GourndTruth after processed
	//static double NPR(const cv::Mat& Result, const cv::Mat& GroundTruth) {
	//	
	//}

	static void Measure(const char* Dir) {
		printf("======== Meausre %s ========\n", Dir);
		std::string MGMMPath = std::string(Dir) + "\\mgmm_mask.png";
		std::string GMMPath = std::string(Dir) + "\\gmm_mask.png";
		std::string GroundTruthPath = std::string(Dir) + "\\ground_truth_mask.png";

		auto MGMMImage = cv::imread(MGMMPath, cv::IMREAD_GRAYSCALE);
		MGMMImage.convertTo(MGMMImage, CV_8UC1);

		auto GroundTruthImage = cv::imread(GroundTruthPath, cv::IMREAD_GRAYSCALE);
		GroundTruthImage.convertTo(GroundTruthImage, CV_8UC1);		
		
		auto GMMImage = cv::imread(GMMPath, cv::IMREAD_GRAYSCALE);
		GMMImage.convertTo(GMMImage, CV_8UC1);

		double VIIndex = VI(MGMMImage, GroundTruthImage);
		double GCEIndex = GCE(MGMMImage, GroundTruthImage);
		double RandIndex = Rand(MGMMImage, GroundTruthImage);
		printf("MGMM\tVI: %.3f, GCE: %.3f, Rand: %.3f\n", VIIndex/*0.f*/, GCEIndex, RandIndex/*0.f*/);

		VIIndex = VI(GMMImage, GroundTruthImage);
		GCEIndex = GCE(GMMImage, GroundTruthImage);
		RandIndex = Rand(GMMImage, GroundTruthImage);
		printf("GMM\tVI: %.3f, GCE: %.3f, Rand: %.3f\n", VIIndex/*0.f*/, GCEIndex, RandIndex/*0.f*/);
	}

	static void Main() {
		Measure("D:\\Study\\毕业设计\\周汇报\\第八周\\output_target\\310007"); // 雪地
		Measure("D:\\Study\\毕业设计\\周汇报\\第八周\\output_target\\296059"); // 象
		Measure("D:\\Study\\毕业设计\\周汇报\\第八周\\output_target\\38092"); // 牛

		Measure("D:\\Study\\毕业设计\\周汇报\\第八周\\output_target\\80099"); // 水牛 
		Measure("D:\\Study\\毕业设计\\周汇报\\第八周\\output_target\\86016"); // 坛
	}
}

