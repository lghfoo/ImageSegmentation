//#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include<cstring>
#define PI 3.141592653589793238463
#define E  2.718281828459045235360
namespace CV {
	static bool ReadImage(const cv::String& InFilename, cv::Mat& OutImage, int flags = cv::IMREAD_COLOR) {
		OutImage = cv::imread(InFilename, flags);
		return !OutImage.empty();
	}

	static void SaveImage(const cv::Mat& InImage, const cv::String& InFilename) {
		cv::imwrite(InFilename, InImage);
	}

	static void DisplayImage(const cv::Mat& Image, const cv::String& WindowName = "Untitled", int WindowFlags = 1) {
		cv::namedWindow(WindowName, WindowFlags);
		cv::moveWindow(WindowName, 0, 0);
		cv::imshow(WindowName, Image);
	}

	static void Wait(int Delay = 0) {
		cv::waitKey(Delay);
	}

	static std::string ToString(const double Input) {
		std::stringstream Stream;
		Stream << Input;
		return Stream.str();
	}

	static std::string ToString(const cv::Vec3d& Vec) {
		std::stringstream Stream;
		const char* Format = "[%.4f, %.4f, %.4f]";
		char Buffer[256];
		memset(Buffer, 0, sizeof(Buffer));
		sprintf_s(Buffer, Format, Vec[0], Vec[1], Vec[2]);
		Stream << Buffer;
		return Stream.str();
	}

	static std::string ToString(const cv::Mat& Mat) {
		const char* Format = "%.4f";
		char Buffer[8];
		std::stringstream Stream;
		Stream << "[";
		for (int i = 0; i < Mat.rows; i++) {
			Stream << "[";
			for (int j = 0; j < Mat.cols; j++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, Mat.at<double>(i, j));
				Stream << Buffer;
				if (j != Mat.cols - 1)Stream << ", ";
			}
			Stream << "]";
		}
		Stream << "]";
		return Stream.str();
	}
}

namespace Math {
	static double MaxValue(const std::vector<double>& Vec) {
		return *std::max_element(Vec.begin(), Vec.end());
	}

	static cv::Vec3d Vec3dSquare(const cv::Vec3d& Input) {
		return cv::Vec3d(Input[0] * Input[0], Input[1] * Input[1], Input[2] * Input[2]);
	}

	static double Vec3dAbs(const cv::Vec3d& Input) {
		return (std::abs(Input[0]) + std::abs(Input[1]) + std::abs(Input[2])) / 3.0;
	};

	static double MatAbs(const cv::Mat& Input) {
		double Sum = 0.0;
		for (int i = 0; i < Input.rows; i++) {
			for (int j = 0; j < Input.cols; j++) {
				Sum += Input.at<double>(i, j);
			}
		}
		return Sum / (double(Input.rows) * Input.cols);
	};

	static double Abs(const cv::Vec3d& Input) {
		return Vec3dAbs(Input);
	}

	static double Abs(const cv::Mat& Input) {
		return MatAbs(Input);
	}

	static double Abs(const double Input) {
		return std::abs(Input);
	}

	static double VarianceSquare(const double Input) {
		return Input * Input;
	}

	static cv::Mat VarianceSquare(const cv::Mat& Input) {
		return Input * Input.t();
	}

	static cv::Mat VarianceSquare(const cv::Vec3d& Input) {
		return VarianceSquare(cv::Mat(Input));
	}

	static void SumVariance(double& Variance, const double Diff, double Scale) {
		Variance += Diff * Diff * Scale;
	}

	static void SumVariance(cv::Mat& VarianceMat, const cv::Vec3d& V, double Scale) {
#define SUM_EQUAL(R, W) VarianceMat.at<double>(R, W) += V[R] * V[W] * Scale
		SUM_EQUAL(0, 0);
		SUM_EQUAL(0, 1);
		SUM_EQUAL(0, 2);

		SUM_EQUAL(1, 0);
		SUM_EQUAL(1, 1);
		SUM_EQUAL(1, 2);

		SUM_EQUAL(2, 0);
		SUM_EQUAL(2, 1);
		SUM_EQUAL(2, 2);
#undef SUM_EQUAL
	}

	static double Square(const double Input) { return Input * Input; }

	static double Square(const cv::Vec3d& Input) {
		return (Input[0] * Input[0] + Input[1] * Input[1] + Input[2] * Input[2]);
	}

	static double Length(const double Input) { return Input; }

	static double Length(const cv::Vec3d& Input) {
		return std::sqrt(Input[0] * Input[0] + Input[1] * Input[1] + Input[2] * Input[2]);
	}

	template<typename T>
	T Empty() { return T(); }

	template<>
	double Empty() { return 0; }

	template<>
	cv::Vec3d Empty() { return { 0,0,0 }; }
}

namespace GMM {

	struct SegArg {
		enum class ESegType { GMMGray, GMMColor, KMeansGray, KMeansColor, MGMMGray, MGMMColor };
		const char* mInputImageName = nullptr;
		int mMaxIterationCount = 20;
		double mDVarThreshold = 0.001;
		double mDExpThreshold = 0.001;
		double mDCoeThreshold = 0.001;
		double mDLogLikehoodThreshold = 100;
		int mComponentCount = 4;
		bool mKMeansInitialized = false; // for GMM & MGMM initialized
		bool mGMMInitialized = false; // for MGMM initialized
		bool mRandomSeed = false;
		ESegType mSegType = ESegType::GMMGray;
		const char* mInputModel = nullptr;
		const char* mOutputModel = nullptr;
		double mBeta = 12.0;
		int mWindSize = 5;
		std::vector<std::vector<uchar>> mColorMap = {};
		SegArg& InputImageName(const char* Name) {
			this->mInputImageName = Name;
			return *this;
		}

		SegArg& MaxIterationCount(int Count) {
			this->mMaxIterationCount = Count;
			return *this;
		}

		SegArg& DVarThreshold(double Threshold) {
			this->mDVarThreshold = Threshold;
			return *this;
		}

		SegArg& DExpThreshold(double Threshold) {
			this->mDExpThreshold = Threshold;
			return *this;
		}

		SegArg& DCoeThreshold(double Threshold) {
			this->mDCoeThreshold = Threshold;
			return *this;
		}

		SegArg& DLogLikehoodThreshold(double Threshold) {
			this->mDLogLikehoodThreshold = Threshold;
			return *this;
		}

		SegArg& ComponentCount(int Count) {
			this->mComponentCount = Count;
			return *this;
		}

		SegArg& KMeansInitialized(bool UseKMeans) {
			this->mKMeansInitialized = UseKMeans;
			return *this;
		}

		SegArg& GMMInitialized(bool UseGMMIni) {
			this->mGMMInitialized = UseGMMIni;
			return *this;
		}

		SegArg& RandomSeed(bool Random) {
			this->mRandomSeed = Random;
			return *this;
		}

		SegArg& SegType(SegArg::ESegType Type) {
			this->mSegType = Type;
			return *this;
		}

		SegArg& InputModel(const char* Model) {
			this->mInputModel = Model;
			return *this;
		}

		SegArg& OutputModel(const char* Model) {
			this->mOutputModel = Model;
			return *this;
		}

		SegArg& Beta(const double Input) {
			mBeta = Input;
			return *this;
		}

		SegArg& WindSize(const int Input) {
			mWindSize = Input;
			return *this;
		}

		SegArg& ColorMap(const std::vector<std::vector<uchar>> C) {
			mColorMap = C;
			return *this;
		}

		bool IsGray() const {
			return this->mSegType == ESegType::GMMGray || this->mSegType == ESegType::MGMMGray || this->mSegType == ESegType::KMeansGray;
		}

		bool IsEM() const {
			return mSegType != ESegType::KMeansGray && mSegType != ESegType::KMeansColor;
		}

		std::string ToString() {
			char ColorMapBuffer[1024] = {};
			std::stringstream Stream;
			Stream << "[";
			for (int i = 0; i < mColorMap.size(); i++) {
				auto C = mColorMap[i];
				std::memset(ColorMapBuffer, 0, sizeof(ColorMapBuffer));
				sprintf(ColorMapBuffer, "[%d, %d, %d]", C[0], C[1], C[2]);
				Stream << ColorMapBuffer;
				if (i != mColorMap.size() - 1)Stream << ", ";
			}
			Stream << "]";

			char Buffer[4096] = {};
			const char* Format =
				"InputImageName: %s\n"
				"MaxIterationCount: %d\n"
				"DVarThreshold: %f\n"
				"DExpThreshold: %f\n"
				"DCoeThreshold: %f\n"
				"DLogLikehoodThreshold: %f\n"
				"ComponentCount: %d\n"
				"KMeansInitialized: %s\n"
				"RandomSeed: %s\n"
				"SegType: %s\n"
				"InputModel: %s\n"
				"OutputModel: %s\n"
				"Beta: %f\n"
				"WindSize: %d\n"
				"ColorMap: %s\n";

			sprintf(Buffer, Format,
				mInputImageName ? mInputImageName : "Null",
				mMaxIterationCount,
				mDVarThreshold,
				mDExpThreshold,
				mDCoeThreshold,
				mDLogLikehoodThreshold,
				mComponentCount,
				mKMeansInitialized ? "True" : "False",
				mRandomSeed ? "True" : "False",
				mSegType == ESegType::GMMColor ? "GMMColor" :
				mSegType == ESegType::GMMGray ? "GMMGray" :
				mSegType == ESegType::KMeansColor ? "KMeansColor" :
				mSegType == ESegType::KMeansGray ? "KMeansGray" :
				mSegType == ESegType::MGMMColor ? "MGMMColor" :
				mSegType == ESegType::MGMMGray ? "MGMMGray" :
				"Unkown SegType",
				mInputModel ? mInputModel : "Null",
				mOutputModel ? mOutputModel : "Null",
				mBeta,
				mWindSize,
				Stream.str().c_str()
			);
			return std::string(Buffer);
		}
	};

	struct SegOutput {
		cv::Mat SegmentedImage;
		int IterationCount = 0;
		std::vector<double>LogLikehoodSummary = {};
		std::string ModelString = "";
	};

	struct GaussianDistribution1D {
		double Expectation = 0;
		double Variance = 0.1;
		double Cache = 0;
		bool UseCache = true;
		double Evaluate(double X) const {
			if (UseCache) {
				return (1.0 / (std::sqrt(2.0 * PI * Variance))) * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
			}
			else {
				return Cache * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
			}
		}
		void UpdateCache() {
			Cache = (1.0 / (std::sqrt(2.0 * PI * Variance)));
		}

		using PixelType = double;
		using VarianceType = double;
		static VarianceType EmptyVariance() {
			return 0;
		}
	};

	struct GaussianDistribution3D {
		cv::Vec3d Expectation = { 0, 0, 0 };
		cv::Mat Variance = (cv::Mat_<double>(3, 3) << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1);
		double Cache1 = 0;
		cv::Mat Cache2;
		bool UseCache = true;
		double Evaluate(const cv::Vec3d& X) const {
			if (!UseCache) {
				auto TmpMat = cv::Mat(X - Expectation).t();
				return 1.0 / std::sqrt(std::pow(2.0 * PI, 3) * cv::determinant(Variance))
					* std::pow(E, -0.5 * (TmpMat * Variance.inv()).dot(TmpMat));
			}
			else {
				return Cache1 * std::pow(E, -0.5 * FastEval(X));
			}
		}
		double FastEval(const cv::Vec3d& X) const {
			//auto TmpMat = cv::Mat(X - Expectation).t();
			//return (TmpMat * Cache2).dot(TmpMat);

			auto V0 = X[0] - Expectation[0],
				V1 = X[1] - Expectation[1],
				V2 = X[2] - Expectation[2];

			auto A00 = Cache2.at<double>(0, 0),
				A01 = Cache2.at<double>(0, 1),
				A02 = Cache2.at<double>(0, 2),
				A10 = Cache2.at<double>(1, 0),
				A11 = Cache2.at<double>(1, 1),
				A12 = Cache2.at<double>(1, 2),
				A20 = Cache2.at<double>(2, 0),
				A21 = Cache2.at<double>(2, 1),
				A22 = Cache2.at<double>(2, 2);

			return V0 * (V0 * A00 + V1 * A10 + V2 * A20)
				+ V1 * (V0 * A01 + V1 * A11 + V2 * A21)
				+ V2 * (V0 * A02 + V1 * A12 + V2 * A22);

			// use avx?

			//return 1;
		}
		void UpdateCache() {
			Cache1 = 1.0 / std::sqrt(std::pow(2.0 * PI, 3) * cv::determinant(Variance));
			Cache2 = Variance.inv();
		}

		using PixelType = cv::Vec3d;
		using VarianceType = cv::Mat;
		static VarianceType EmptyVariance() {
			return (cv::Mat_<double>(3, 3) << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1);
		}
	};

	struct EMContext {
		int IterationCount = 0;
		std::vector<double>DExp;
		std::vector<double>DVar;
		std::vector<double>DCoeff;
		double DLogLikehood;
		double LastLogLikehood = 0;
		bool HasLastLogLikehood = false;
		std::vector<double> LogLikehoodSummary = {};

		EMContext(int K) :
			DCoeff(std::vector<double>(K, DBL_MAX)),
			DExp(std::vector<double>(K, DBL_MAX)),
			DVar(std::vector<double>(K, DBL_MAX)),
			DLogLikehood(DBL_MAX) {
		}

		double MaxDExp() const {
			return Math::MaxValue(DExp);
		}

		double MaxDVar() const {
			return Math::MaxValue(DVar);
		}

		double MaxDCoeff() const {
			return Math::MaxValue(DCoeff);
		}

		std::string ToString() const {
			std::stringstream Stream;
			Stream << "Iteration\t: " << IterationCount << "\n"
				<< "Max DCoeff\t: " << MaxDCoeff() << "\n"
				<< "Max DExp\t: " << MaxDExp() << "\n"
				<< "Max DVar\t: " << MaxDVar() << "\n"
				<< "DLogLikehood\t: " << DLogLikehood << "\n";
			return Stream.str();
		}
	};

	struct EMStopCondition
	{
		int mMaxIterationCount = 20;
		double mDVarThreshold = 0.001;
		double mDExpThreshold = 0.001;
		double mDCoeThreshold = 0.001;
		double mDLogLikehoodThreshold = 100;

		bool IsSatisfied(const EMContext& Context) const {
			return Context.IterationCount >= mMaxIterationCount
				|| (Context.MaxDExp() <= mDExpThreshold
					&& Context.MaxDCoeff() <= mDCoeThreshold
					&& Context.MaxDVar() <= mDVarThreshold
					&& Context.DLogLikehood <= mDLogLikehoodThreshold);
		}

		EMStopCondition& MaxIterationCount(int Count) {
			this->mMaxIterationCount = Count;
			return *this;
		}

		EMStopCondition& DVarThreshold(double Threshold) {
			this->mDVarThreshold = Threshold;
			return *this;
		}

		EMStopCondition& DExpThreshold(double Threshold) {
			this->mDExpThreshold = Threshold;
			return *this;
		}

		EMStopCondition& DCoeThreshold(double Threshold) {
			this->mDCoeThreshold = Threshold;
			return *this;
		}

		EMStopCondition& DLogLikehoodThreshold(double Threshold) {
			this->mDLogLikehoodThreshold = Threshold;
			return *this;
		}
	};

	struct MixtureModel {
		EMContext Context;
		EMStopCondition Condition;
		std::vector<cv::Mat> PostProbability;
		MixtureModel(int K) :
			Context(EMContext(K)),
			PostProbability(std::vector<cv::Mat>(K)) {}
		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize, bool UseGMMInitialize = false) = 0;
		virtual void EStep(const cv::Mat& InImage) = 0;
		virtual void MStep(const cv::Mat& InImage) = 0;
		virtual std::string TypeString() const = 0;
		virtual std::string ToString() const = 0;
		virtual void Save(const char* Filename) const = 0;
		virtual void Load(const char* Filename) = 0;
	};

	struct EMAlgorithm {
		static void Train(MixtureModel& Model, const cv::Mat& InImage, bool UseKMeansInitialize = true, bool UseGMMInitialize = false) {
			Model.Initialize(InImage, UseKMeansInitialize, UseGMMInitialize);
			auto& Context = Model.Context;
			auto& Condition = Model.Condition;
			printf("**************** Train %s ****************\n", Model.TypeString().c_str());
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
			// 迭代求解
			while (!Condition.IsSatisfied(Context)) {
				printf("======== Iter #%d ========\n", Context.IterationCount + 1);
				// E-Step
				Model.EStep(InImage);
				// M-Step
				Model.MStep(InImage);
				// Update Context
				Context.IterationCount++;
				printf("Model: \n%s", Model.ToString().c_str());
				printf("Context: \n%s\n", Context.ToString().c_str());
			}
			printf("-------- ModifiedGMMSegmentation End --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
		}
	};

	template<typename PixelType>
	struct Cluster {
		PixelType Sum = Math::Empty<PixelType>();
		int Count = 0;
		void Add(const PixelType& Sample) {
			Sum += Sample;
			Count++;
		}
		PixelType Center() {
			assert(Count > 0);
			return Sum / Count;
		}
	};

	template<typename PixelType>
	static void KMeansInitialize(const cv::Mat& InImage, const int K, std::vector<PixelType>& OutMeans) {
		printf("-------- K-Means --------\n");
		OutMeans.resize(K);
		// 随机初始化
		for (int i = 0; i < K; i++) {
			int RandRow = std::rand() % InImage.rows;
			int RandCol = std::rand() % InImage.cols;
			OutMeans[i] = InImage.at<PixelType>(RandRow, RandCol);
		}

		std::vector<Cluster<PixelType>>Clusteres(K);
		int N = InImage.rows * InImage.cols;
		bool HasMeansUpdated = false;
		//// K-Means
		do {
			std::fill(Clusteres.begin(), Clusteres.end(), Cluster<PixelType>());
			// a) 将样本划入簇
			for (int i = 0; i < N; i++) {
				auto Sample = InImage.at<PixelType>(i);

				double MinError = 0;
				int MinIndex = 0;
				for (int j = 0; j < K; j++) {
					auto Error = Math::Square(Sample - OutMeans[j]);
					if (j == 0 || Error < MinError) {
						MinError = Error;
						MinIndex = j;
					}
				}
				Clusteres[MinIndex].Add(Sample);
			}
			// b) 更新均值向量
			double Threshold = 1e-8;
			HasMeansUpdated = false;
			printf("================ Update Means ================\n");
			for (int i = 0; i < K; i++) {
				if (Clusteres[i].Count == 0) {
					printf("Diff #%d: cluster element count is 0\n", i);
					continue;
				}
				double Diff = Math::Abs(OutMeans[i] - Clusteres[i].Center());
				printf("Diff #%d: %.6f\n", i, Diff);
				if (Diff > Threshold) {
					OutMeans[i] = Clusteres[i].Center();
					HasMeansUpdated = true;
				}
			}
		} while (HasMeansUpdated);
	}

	static void KMeansGrayInitialize(const cv::Mat& InImage, const int K, std::vector<double>& OutMeans) {
		KMeansInitialize<double>(InImage, K, OutMeans);
	}

	static void KMeansColorInitialize(const cv::Mat& InImage, const int K, std::vector<cv::Vec3d>& OutMeans) {
		KMeansInitialize<cv::Vec3d>(InImage, K, OutMeans);
	}

	struct ModelFileManager {
		template<typename GaussianType>
		static void Save(const char* Filename, const std::vector<double>& MixtureCoefficients,
			const std::vector<GaussianType>& GaussianDistributions) {
			FILE* File = fopen(Filename, "wb");
			auto DistSize = GaussianDistributions.size();
			fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
			fwrite(&GaussianDistributions[0], sizeof(GaussianType), DistSize, File);
			auto MixSize = MixtureCoefficients.size();
			fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
			fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
			fclose(File);
		}

		template<typename GaussianType>
		static void Load(const char* Filename, std::vector<double>& MixtureCoefficients, std::vector<GaussianType>& GaussianDistributions) {
			FILE* File = fopen(Filename, "rb");
			size_t DistSize = 0, MixSize = 0;

			fread(&DistSize, sizeof(size_t), 1, File);
			GaussianDistributions.resize(DistSize);
			fread(&GaussianDistributions[0], sizeof(GaussianType), DistSize, File);

			fread(&MixSize, sizeof(size_t), 1, File);
			MixtureCoefficients.resize(MixSize);
			fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

			fclose(File);
		}
	};

	struct Initializer {
		template<typename GaussianType>
		static void Initialize(bool UseKMeansInitialize,
			const cv::Mat& InImage, const int K,
			std::vector<GaussianType>& GaussianDistributions,
			std::vector<double>& MixtureCoefficients,
			std::vector<cv::Mat>& PostProbability
		) {
			printf("Initialize....\n");
			if (UseKMeansInitialize) {
				std::vector<GaussianType::PixelType>Means;
				KMeansInitialize<GaussianType::PixelType>(InImage, K, Means);
				for (int i = 0; i < K; i++) {
					GaussianDistributions.at(i).Expectation = Means[i];
				}
			}
			else {
				for (int i = 0; i < K; i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GaussianDistributions.at(i).Expectation = InImage.at<GaussianType::PixelType>(RandRow, RandCol);
				}
			}

			std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

			for (int i = 0; i < PostProbability.size(); i++) {
				PostProbability[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
			}
		}
	};

	struct GaussianMixtureModelEMStepper {
		template<typename GaussianType>
		static void EStep(const cv::Mat& InImage,
			const int K,
			std::vector<GaussianType>& GaussianDistributions,
			const std::vector<double>& MixtureCoefficients,
			std::vector<cv::Mat>& PostProbability) {
			// update cache
			for (int i = 0; i < K; i++) {
				GaussianDistributions.at(i).UpdateCache();
			}
			for (int i = 0; i < K; i++) {
				for (int row = 0; row < PostProbability[i].rows; row++) {
#pragma omp parallel for
					for (int col = 0; col < PostProbability[i].cols; col++) {
						double Up = MixtureCoefficients.at(i) * GaussianDistributions.at(i).Evaluate(InImage.at<typename GaussianType::PixelType>(row, col));
						double Sum = 0;
						for (int j = 0; j < K; j++) {
							Sum += MixtureCoefficients.at(j) * GaussianDistributions.at(j).Evaluate(InImage.at<typename GaussianType::PixelType>(row, col));
						}
						PostProbability[i].at<double>(row, col) = Up / Sum;
					}
				}
			}
		}

		template<typename GaussianType>
		static void MStep(const cv::Mat& InImage,
			const int K, std::vector<GaussianType>& GaussianDistributions,
			std::vector<double>& MixtureCoefficients,
			const std::vector<cv::Mat>& PostProbability,
			EMContext& Context) {
			auto N = InImage.rows * InImage.cols;
#pragma omp parallel for
			for (int i = 0; i < K; i++) {
				double SumProbility = 0.0;
				GaussianType::PixelType SumExpectation = 0.0;
				for (int row = 0; row < PostProbability[i].rows; row++) {
					for (int col = 0; col < PostProbability[i].cols; col++) {
						SumProbility += PostProbability[i].at<double>(row, col);
						SumExpectation += PostProbability[i].at<double>(row, col) * InImage.at<GaussianType::PixelType>(row, col);
					}
				}
				auto& OldCoeff = MixtureCoefficients.at(i);
				auto& OldGaussianDistrib = GaussianDistributions.at(i);
				auto NewCoeff = SumProbility / N;
				auto NewGaussianDistribExpectation = SumExpectation / SumProbility;

				auto SumVariance = GaussianType::EmptyVariance();
				for (int row = 0; row < PostProbability[i].rows; row++) {
					for (int col = 0; col < PostProbability[i].cols; col++) {
						auto Diff = InImage.at<GaussianType::PixelType>(row, col) - NewGaussianDistribExpectation;
						//SumVariance += PostProbability[i].at<double>(row, col) * Math::VarianceSquare(Diff);
						Math::SumVariance(SumVariance, Diff, PostProbability[i].at<double>(row, col));
					}
				}
				auto NewGaussianDistribVariance = SumVariance / SumProbility;

				Context.DCoeff[i] = std::abs(OldCoeff - NewCoeff);
				Context.DExp[i] = Math::Abs(OldGaussianDistrib.Expectation - NewGaussianDistribExpectation);
				Context.DVar[i] = Math::Abs(OldGaussianDistrib.Variance - NewGaussianDistribVariance);

				OldCoeff = NewCoeff;
				OldGaussianDistrib.Expectation = NewGaussianDistribExpectation;
				OldGaussianDistrib.Variance = NewGaussianDistribVariance;
			}

			// Compute LogLikehood
			auto LogLikehood = 0.0;
			for (int i = 0; i < N; i++) {
				double Sum = 0.0;
				for (int k = 0; k < K; k++) {
					Sum += MixtureCoefficients.at(k) * GaussianDistributions.at(k).Evaluate(InImage.at<GaussianType::PixelType>(i));
				}
				LogLikehood += std::log(Sum);
			}

			if (Context.HasLastLogLikehood) {
				Context.DLogLikehood = std::abs(LogLikehood - Context.LastLogLikehood);
			}
			else {
				Context.HasLastLogLikehood = true;
			}
			Context.LastLogLikehood = LogLikehood;
			Context.LogLikehoodSummary.push_back(LogLikehood);
		}
	};

	struct GaussianMixtureModel1D : MixtureModel
	{
		using DistributionType = GaussianDistribution1D;
		int K;
		std::vector<GaussianDistribution1D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		GaussianMixtureModel1D(const int K) :
			MixtureModel(K),
			K(K),
			GaussianDistributions(std::vector<GaussianDistribution1D>(K)),
			MixtureCoefficients(std::vector<double>(K)) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true, bool UseGMMInitialize = false) override {
			Initializer::Initialize<GaussianDistribution1D>(UseKMeansInitialize, InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			GaussianMixtureModelEMStepper::EStep<GaussianDistribution1D>(
				InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability);
		}

		virtual void MStep(const cv::Mat& InImage) override {
			GaussianMixtureModelEMStepper::MStep<GaussianDistribution1D>(
				InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability, Context
				);
		}

		virtual std::string TypeString() const override {
			return "GaussianMixtureModel 1D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Coe: %.6f\tExp: %.6f\tVar: %.6f\n";
			char Buffer[256];
			for (int i = 0; i < K; i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, MixtureCoefficients[i], GaussianDistributions[i].Expectation, GaussianDistributions[i].Variance);
				Stream << Buffer;
			}
			Stream << "\n";
			return Stream.str();
		}

		virtual void Save(const char* Filename) const override {
			ModelFileManager::Save<GaussianDistribution1D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}

		virtual void Load(const char* Filename) override {
			ModelFileManager::Load<GaussianDistribution1D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}
	};

	struct GaussianMixtureModel3D : MixtureModel
	{
		using DistributionType = GaussianDistribution3D;
		int K;
		std::vector<GaussianDistribution3D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		GaussianMixtureModel3D(const int K) :
			MixtureModel(K),
			K(K),
			GaussianDistributions(std::vector<GaussianDistribution3D>(K)),
			MixtureCoefficients(std::vector<double>(K)) {
		}


		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true, bool UseGMMInitialize = false) override {
			Initializer::Initialize<GaussianDistribution3D>(UseKMeansInitialize, InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			GaussianMixtureModelEMStepper::EStep<GaussianDistribution3D>(
				InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability);
		}

		virtual void MStep(const cv::Mat& InImage) override {
			GaussianMixtureModelEMStepper::MStep<GaussianDistribution3D>(
				InImage, K, GaussianDistributions,
				MixtureCoefficients, PostProbability, Context
				);
		}

		virtual std::string TypeString() const override {
			return "GaussianMixtureModel 3D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Coe: %.6f\tExp: %s\tVar: %s\n";
			char Buffer[256];
			for (int i = 0; i < K; i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, MixtureCoefficients[i],
					CV::ToString(GaussianDistributions[i].Expectation).c_str(),
					CV::ToString(GaussianDistributions[i].Variance).c_str());
				Stream << Buffer;
			}
			Stream << "\n";
			return Stream.str();
		}

		virtual void Save(const char* Filename) const override {
			ModelFileManager::Save<GaussianDistribution3D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}

		virtual void Load(const char* Filename) override {
			ModelFileManager::Load<GaussianDistribution3D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}
	};

	struct ModifiedGaussianMixtureModelEMStepper {
		template<typename GaussianType>
		static void UpdateCache(std::vector<GaussianType>& GaussianDistributions) {
			for (auto& Gaussian : GaussianDistributions) {
				Gaussian.UpdateCache();
			}
		}

		template<typename GaussianType>
		static void EStep(const cv::Mat& InImage,
			const int K, const int N,
			const std::vector<double>& MixtureCoefficients,
			std::vector<GaussianType>& GaussianDistributions,
			std::vector<cv::Mat>& PostProbability
		) {
			UpdateCache(GaussianDistributions);
			printf(">>>>>>>>>>>>>>>> E-Step....\n");
			for (int i = 0; i < K; i++) {
				printf("Estimate %d/%d\n", i + 1, K);
				//auto Start = std::clock();
#pragma omp parallel for
				for (int NIndex = 0; NIndex < N; NIndex++) {
					double Up = MixtureCoefficients.at(i * size_t(N) + NIndex) * GaussianDistributions.at(i).Evaluate(InImage.at<GaussianType::PixelType>(NIndex));
					double Sum = 1e-9;
					for (int j = 0; j < K; j++) {
						Sum += MixtureCoefficients.at(j * size_t(N) + NIndex) * GaussianDistributions.at(j).Evaluate(InImage.at<GaussianType::PixelType>(NIndex));
					}
					PostProbability[i].at<double>(NIndex) = Up / Sum;
				}
				//auto End = std::clock();
				//printf("use %d ms\n", (End - Start));
			}
		}

		template<typename GaussianType>
		static void MStep(const cv::Mat& InImage,
			const int K, const int N,
			const std::vector<cv::Mat>& PostProbability,
			const double Beta,
			const int WindSize,
			EMContext& Context,
			std::vector<double>& MixtureCoefficients,
			std::vector<GaussianType>& GaussianDistributions,
			std::vector<double>& GCache
		) {
			printf(">>>>>>>>>>>>>>>> M-Step....\n");
			// 计算上一次迭代的Gij
			{
				printf("Compute GCache....\n");
#pragma omp parallel for
				for (int i = 0; i < InImage.rows; i++) {
					for (int j = 0; j < InImage.cols; j++) {
						for (int k = 0; k < K; k++) {
							int NIndex = i * InImage.cols + j;
							GCache[k * size_t(N) + NIndex] = G(k, NIndex, InImage.rows, InImage.cols,
								N, PostProbability, MixtureCoefficients, Beta, WindSize);
						}
					}
				}
			}

			// 更新每个高斯分布
			{
				printf("Update Gaussian Parameters....\n");
				//auto Start = std::clock();
#pragma omp parallel for
				for (int i = 0; i < K; i++) {
					double SumProbility = 0.0;
					GaussianType::PixelType SumExpectation = Math::Empty<GaussianType::PixelType>();
					for (int row = 0; row < PostProbability[i].rows; row++) {
						for (int col = 0; col < PostProbability[i].cols; col++) {
							SumProbility += PostProbability[i].at<double>(row, col);
							SumExpectation += PostProbability[i].at<double>(row, col) * InImage.at<GaussianType::PixelType>(row, col);
						}
					}

					auto& OldGaussianDistrib = GaussianDistributions.at(i);
					GaussianType NewGaussianDistrib;

					// 期望
					NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

					// 方差
					auto SumVariance = GaussianType::EmptyVariance();
					auto Diff = Math::Empty<GaussianType::PixelType>();
					auto& PP = PostProbability[i];
					for (int i = 0; i < N; i++) {
						Diff = InImage.at<GaussianType::PixelType>(i) - NewGaussianDistrib.Expectation;
						//SumVariance += PP.at<double>(i) *  Math::VarianceSquare(Diff);
						Math::SumVariance(SumVariance, Diff, PP.at<double>(i));
					}

					NewGaussianDistrib.Variance = SumVariance / SumProbility;

					Context.DExp[i] = Math::Abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
					Context.DVar[i] = Math::Abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);
					Context.DCoeff[i] = 0;

					OldGaussianDistrib = NewGaussianDistrib;
				}
				//auto End = std::clock();
				//printf("use %d ms\n", End - Start);
			}

			// 更新系数
			{
				printf("Update Coefficient....\n");
#pragma omp parallel for
				for (int i = 0; i < InImage.rows; i++) {
					for (int j = 0; j < InImage.cols; j++) {
						int NIndex = i * InImage.cols + j;
						double SumZG = 0.0;
						for (int k = 0; k < K; k++) {
							SumZG += (PostProbability[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols, N, PostProbability, MixtureCoefficients,
									Beta, WindSize));
						}

						for (int k = 0; k < K; k++) {
							auto ZG = (PostProbability[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols, N, PostProbability, MixtureCoefficients,
									Beta, WindSize));
							MixtureCoefficients.at(k * size_t(N) + NIndex) = ZG / SumZG;
						}
					}
				}
			}

			// 计算似然函数
			{
				printf("Compute LogLikehood: ");
				UpdateCache(GaussianDistributions);
				// first term
				double FirstTerm = 0;
#pragma omp parallel for reduction (+:FirstTerm)
				for (int i = 0; i < N; i++) {
					double SumTemp = 1e-9;
					for (int k = 0; k < K; k++) {
						SumTemp += MixtureCoefficients.at(k * size_t(N) + i)
							* GaussianDistributions[k].Evaluate(InImage.at<GaussianType::PixelType>(i));
					}
					FirstTerm += std::log(SumTemp);
				}
				// second term
				double SecondTerm = 0;
#pragma omp parallel for reduction (+:SecondTerm)
				for (int i = 0; i < N; i++) {
					double SumTemp = 1e-9;
					for (int k = 0; k < K; k++) {
						SumTemp += GCache[k * size_t(N) + i]
							* std::log(MixtureCoefficients.at(k * size_t(N) + i));
					}
					SecondTerm += SumTemp;
				}

				auto LogLikehood = FirstTerm + SecondTerm;
				if (Context.HasLastLogLikehood) {
					Context.DLogLikehood = std::abs(LogLikehood - Context.LastLogLikehood);
				}
				else {
					Context.HasLastLogLikehood = true;
				}
				Context.LastLogLikehood = LogLikehood;
				Context.LogLikehoodSummary.push_back(LogLikehood);
			}
		}

		static double G(const int KIndex, const int NIndex, const int Rows, const int Cols, const int N,
			const std::vector<cv::Mat>& PostProbability,
			const std::vector<double>& MixtureCoefficients,
			const double Beta,
			const int WindSize
		) {
			const int Ni = WindSize * WindSize;
			double Sum = 0.0;
			int Row = NIndex / Cols;
			int Col = NIndex % Cols;
			for (int i = -WindSize / 2; i <= WindSize / 2; i++) {
				for (int j = -WindSize / 2; j <= WindSize / 2; j++) {
					int R = Row + i, C = Col + j;
					double Z = 0, M = 0;
					if (0 <= R && R < Rows && 0 <= C && C < Cols) {
						Z = PostProbability[KIndex].at<double>(R, C);
						M = MixtureCoefficients.at(KIndex * size_t(N) + R * size_t(Cols) + C);
					}
					Sum += (Z + M);
				}
			}
			return std::pow(E, (Beta / (2.0 * Ni)) * Sum);
		}
	};

	struct ModifiedInitializer {
		template<typename GaussianType>
		static void Initialize(
			const cv::Mat& InImage,
			const int K, const int N,
			const EMStopCondition& Condition,
			const bool UseKMeansInitialize,
			std::vector<typename GaussianType::DistributionType>& GaussianDistributions,
			std::vector<double>& MixtureCoefficients,
			std::vector<cv::Mat>& PostProbability
		) {
			printf("Use GMM Initialize....\n");
			GaussianType* Model = new GaussianType(K);
			Model->Condition = Condition;
			EMAlgorithm::Train(*Model, InImage, UseKMeansInitialize, false);
			GaussianDistributions = Model->GaussianDistributions;
			for (int i = 0; i < K; i++) {
				for (int j = 0; j < N; j++) {
					MixtureCoefficients.at(i * size_t(N) + j) = Model->MixtureCoefficients.at(i);
				}
			}
			for (int i = 0; i < PostProbability.size(); i++) {
				PostProbability[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
			}
			delete Model;
		}
	};

	struct ModifiedGaussianMixtureModel1D : MixtureModel
	{
		int K, N;
		std::vector<double>GCache;
		std::vector<GaussianDistribution1D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		double Beta = 12;
		int WindSize = 5;
		ModifiedGaussianMixtureModel1D(const int K, const int N, const double Beta = 12, const int WindSize = 5) :
			MixtureModel(K),
			K(K), N(N), Beta(Beta), WindSize(WindSize),
			MixtureCoefficients(std::vector<double>(K* size_t(N))),
			GaussianDistributions(std::vector<GaussianDistribution1D>(K)),
			GCache(std::vector<double>(K* size_t(N))) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true, bool UseGMMInitialize = false) override {
			if (UseGMMInitialize) {
				ModifiedInitializer::Initialize<GaussianMixtureModel1D>(InImage, K, N, Condition, UseKMeansInitialize, GaussianDistributions, 
					MixtureCoefficients, PostProbability);
			}
			else {
				Initializer::Initialize<GaussianDistribution1D>(UseKMeansInitialize, InImage, K, GaussianDistributions,
					MixtureCoefficients, PostProbability);
			}
		}

		virtual void EStep(const cv::Mat& InImage) override {
			ModifiedGaussianMixtureModelEMStepper::EStep<GaussianDistribution1D>(
				InImage, K, N, MixtureCoefficients,
				GaussianDistributions, PostProbability
				);
		}

		virtual void MStep(const cv::Mat& InImage) override {
			ModifiedGaussianMixtureModelEMStepper::MStep<GaussianDistribution1D>(
				InImage, K, N, PostProbability, Beta, WindSize, Context, MixtureCoefficients,
				GaussianDistributions, GCache
				);
		}

		virtual std::string TypeString() const override {
			return "ModifiedGaussianMixtureModel 1D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Exp: %.6f\tVar: %.6f\n";
			char Buffer[256];
			for (int i = 0; i < K; i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, GaussianDistributions[i].Expectation, GaussianDistributions[i].Variance);
				Stream << Buffer;
			}
			return Stream.str();
		}

		virtual void Save(const char* Filename) const override {
			ModelFileManager::Save<GaussianDistribution1D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}

		virtual void Load(const char* Filename) override {
			ModelFileManager::Load<GaussianDistribution1D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}
	};

	struct ModifiedGaussianMixtureModel3D : MixtureModel
	{
		int K, N;
		std::vector<double> MixtureCoefficients;
		std::vector<GaussianDistribution3D> GaussianDistributions;
		std::vector<double>GCache;
		double Beta = 12;
		int WindSize = 5;
		ModifiedGaussianMixtureModel3D(const int K, const int N, const double Beta = 12, const int WindSize = 5) :
			MixtureModel(K),
			K(K), N(N), Beta(Beta), WindSize(WindSize),
			GaussianDistributions(std::vector<GaussianDistribution3D>(K)),
			MixtureCoefficients(std::vector<double>(K* size_t(N))),
			GCache(std::vector<double>(K* size_t(N))) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true, bool UseGMMInitialize = false) override {
			if (UseGMMInitialize) {
				ModifiedInitializer::Initialize<GaussianMixtureModel3D>(InImage, K, N, Condition, UseKMeansInitialize, GaussianDistributions,
					MixtureCoefficients, PostProbability);
			}
			else {
				Initializer::Initialize<GaussianDistribution3D>(UseKMeansInitialize, InImage, K, GaussianDistributions,
					MixtureCoefficients, PostProbability);
			}
		}

		virtual void EStep(const cv::Mat& InImage) override {
			ModifiedGaussianMixtureModelEMStepper::EStep<GaussianDistribution3D>(
				InImage, K, N, MixtureCoefficients,
				GaussianDistributions, PostProbability
				);
		}

		virtual void MStep(const cv::Mat& InImage) override {
			ModifiedGaussianMixtureModelEMStepper::MStep<GaussianDistribution3D>(
				InImage, K, N, PostProbability, Beta, WindSize, Context, MixtureCoefficients,
				GaussianDistributions, GCache
				);
		}

		virtual std::string TypeString() const override {
			return "ModifiedGaussianMixtureModel 3D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Exp: %s\tVar: %s\n";
			char Buffer[256];
			for (int i = 0; i < K; i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i,
					CV::ToString(GaussianDistributions[i].Expectation).c_str(),
					CV::ToString(GaussianDistributions[i].Variance).c_str());
				Stream << Buffer;
			}
			return Stream.str();
		}

		virtual void Save(const char* Filename) const override {
			ModelFileManager::Save<GaussianDistribution3D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}

		virtual void Load(const char* Filename) override {
			ModelFileManager::Load<GaussianDistribution3D>(Filename, this->MixtureCoefficients, this->GaussianDistributions);
		}
	};

	template<typename PixelType>
	static void KMeansSegmentation(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		std::vector<PixelType> Means;
		KMeansInitialize<PixelType>(InImage, Arg.mComponentCount, Means);
		printf("-------- Segmentation --------\n");
		printf("Means: [");
		for (int i = 0; i < Means.size(); i++) {
			printf("%s", CV::ToString(Means[i]).c_str());
			if (i != Means.size() - 1)printf(", ");
			else printf("]\n");
		}
		int N = InImage.rows * InImage.cols;
		double Step = 1.0 / (double(K) - 1);
		OutImage.forEach<cv::Vec3b>(
			[&](cv::Vec3b& Pixel, const int* Position) {
			double MinError = 0, MinIndex = 0;
			for (int j = 0; j < K; j++) {
				auto Error = Math::Square(InImage.at<PixelType>(Position) - Means[j]);
				if (j == 0 || Error < MinError) {
					MinError = Error;
					MinIndex = j;
				}
			}
			auto& ColorArr = Arg.mColorMap[MinIndex];
			Pixel = { ColorArr[2], ColorArr[1], ColorArr[0] };// Step* MinIndex;
		});
	}

	static void KMeansSegmentationGray(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		KMeansSegmentation<double>(InImage, OutImage, Arg);
	}

	static void KMeansSegmentationColor(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		KMeansSegmentation<cv::Vec3d>(InImage, OutImage, Arg);
	}

	static void Segmentation(const cv::Mat& InImage, const SegArg& Arg, SegOutput& Output) {

		Output.SegmentedImage = cv::Mat(InImage.rows, InImage.cols, CV_8UC3);

		if (Arg.mRandomSeed) {
			std::srand(time(NULL));
		}

		if (Arg.IsEM()) {
			MixtureModel* Model = nullptr;
			switch (Arg.mSegType)
			{
			case SegArg::ESegType::GMMColor:
				Model = new GaussianMixtureModel3D(Arg.mComponentCount);
				break;
			case SegArg::ESegType::GMMGray:
				Model = new GaussianMixtureModel1D(Arg.mComponentCount);
				break;
			case SegArg::ESegType::MGMMColor:
				Model = new ModifiedGaussianMixtureModel3D(Arg.mComponentCount, InImage.rows * InImage.cols, Arg.mBeta, Arg.mWindSize);
				break;
			case SegArg::ESegType::MGMMGray:
				Model = new ModifiedGaussianMixtureModel1D(Arg.mComponentCount, InImage.rows * InImage.cols, Arg.mBeta, Arg.mWindSize);
				break;
			default:
				break;
			}

			if (Model) {
				Model->Condition
					.MaxIterationCount(Arg.mMaxIterationCount)
					.DCoeThreshold(Arg.mDCoeThreshold)
					.DExpThreshold(Arg.mDExpThreshold)
					.DVarThreshold(Arg.mDVarThreshold)
					.DLogLikehoodThreshold(Arg.mDLogLikehoodThreshold);

				EMAlgorithm::Train(*Model, InImage, Arg.mKMeansInitialized, Arg.mGMMInitialized);
				////////////////// Segmenation ////////////////
				Model->EStep(InImage);
				double Step = 1.0 / (double(Arg.mComponentCount) - 1);
				Output.SegmentedImage.forEach<cv::Vec3b>(
					[&](cv::Vec3b& Pixel, const int* Position) {
					double MaxProbility = 0.0;
					int MaxI = 0;
					auto& Probility = Model->PostProbability;
					for (int i = 0; i < Probility.size(); i++) {
						if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
							MaxI = i;
							MaxProbility = Probility[i].at<double>(Position);
						}
					}
					auto& ColorArr = Arg.mColorMap[MaxI];
					Pixel = { ColorArr[2], ColorArr[1], ColorArr[0] };// Step* MaxI;
				}
				);
				Output.LogLikehoodSummary = Model->Context.LogLikehoodSummary;
				Output.ModelString = Model->ToString();
				Output.IterationCount = Model->Context.IterationCount;
				delete Model;
			}
		}
		else {
			switch (Arg.mSegType)
			{
			case SegArg::ESegType::KMeansColor:
				KMeansSegmentationColor(InImage, Output.SegmentedImage, Arg);
				break;
			case SegArg::ESegType::KMeansGray:
				KMeansSegmentationGray(InImage, Output.SegmentedImage, Arg);
				break;
			default:
				break;
			}
		}
	}

	static bool Segmentation(const SegArg& Arg, SegOutput& Output) {
		bool OK = true;
		cv::Mat InputImage;

		if (Arg.IsGray()) {
			OK = CV::ReadImage(Arg.mInputImageName, InputImage, cv::IMREAD_GRAYSCALE);
			InputImage.convertTo(InputImage, CV_64FC1, 1.0 / 255.0);
		}
		else {
			OK = CV::ReadImage(Arg.mInputImageName, InputImage, cv::IMREAD_COLOR);
			InputImage.convertTo(InputImage, CV_64FC3, 1.0 / 255.0);
		}

		if (!OK) {
			printf("read image failed\n");
			return false;
		}

		Segmentation(InputImage, Arg, Output);

		return true;
	}

	static void TestSegmentation(const SegArg& Arg) {
		SegOutput Output;
		auto OK = Segmentation(Arg, Output);

		if (!OK) {
			printf("read image fail\n");
		}
		else {
			cv::Mat InputImage;

			if (Arg.IsGray()) {
				OK = CV::ReadImage(Arg.mInputImageName, InputImage, cv::IMREAD_GRAYSCALE);
				InputImage.convertTo(InputImage, CV_64FC1, 1.0 / 255.0);
			}
			else {
				OK = CV::ReadImage(Arg.mInputImageName, InputImage, cv::IMREAD_COLOR);
				InputImage.convertTo(InputImage, CV_64FC3, 1.0 / 255.0);
			}

			std::stringstream Stream;
			static int Count = 0;
			Stream << "Origin " << Count;
			CV::DisplayImage(InputImage, Stream.str().c_str());
			Stream.clear();
			Stream = std::stringstream();
			Stream << "Segmented" << Count;
			CV::DisplayImage(Output.SegmentedImage, Stream.str().c_str());

			//auto InputName = std::string(InputImageName);
			//auto Pos = InputName.find_last_of('\\');
			//auto SimpleName = InputName.substr(Pos, InputName.size() - Pos);
			//Pos = SimpleName.find('.');
			//auto SavedName = SimpleName.replace(Pos, SimpleName.size(), "_segmented.png");
			//CV::SaveImage(SegmentedImg, SavedName);

			Count++;
		}
	}

	static void Main() {
		std::vector<std::tuple<const char*, int, std::vector<std::vector<uchar>>>>TestData{
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\blog\\002\\0.png", 4, {}						},
			{"D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f02970.png", 4,{}	},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\grid.PNG", 4,{}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\Lenna.jpg", 4,{}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\leaf.jpg", 2, {}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\snow.PNG", 3, {}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\cow.PNG", 4, {}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\elephant.PNG", 2, {}					},
			{"D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f00300.png", 4, {}	},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\horse2.PNG", 2, {}						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\horse1.PNG", 2, {/*background*/{159, 181, 98}, /*horse*/{101, 84, 66}}						},
		};

		auto Arg = SegArg()
			.RandomSeed(false)
			.Beta(12)
			.WindSize(5)
			.InputModel(nullptr)
			.DCoeThreshold(0.001)
			.DExpThreshold(0.001)
			.DVarThreshold(0.001)
			.OutputModel(nullptr)
			.MaxIterationCount(20)
			.KMeansInitialized(true)
			.DLogLikehoodThreshold(1)
			.ColorMap(std::get<2>(TestData.back()))
			.ComponentCount(std::get<1>(TestData.back()))
			.InputImageName(std::get<0>(TestData.back()));

		TestSegmentation(
			Arg
			.SegType(SegArg::ESegType::KMeansColor)
			.SegType(SegArg::ESegType::KMeansGray)
			.SegType(SegArg::ESegType::MGMMGray)
			.SegType(SegArg::ESegType::GMMColor)
			.SegType(SegArg::ESegType::MGMMColor)
			.SegType(SegArg::ESegType::GMMGray)
		);

		//TestSegmentation(
		//	TestData.back().first,
		//	Arg.SegType(SegArg::ESegType::KMeansGray)
		//);

		CV::Wait();
	}
}

// C Interface
#define GMM_API __declspec(dllexport)
extern "C" {
	//////////////// PROTOTYPE ////////////////
	GMM_API struct Color {
		uchar R = 0, G = 0, B = 0;
	};

	GMM_API struct ColorMap {
		int Count = 0;
		Color* Colors = nullptr;
	};

	GMM_API struct SegArg
	{
		//enum class ESegType { GMMGray, GMMColor, KMeansGray, KMeansColor, MGMMGray, MGMMColor };
		const char* InputImageName = nullptr;
		int mMaxIterationCount = 20;
		double mDVarThreshold = 0.001;
		double mDExpThreshold = 0.001;
		double mDCoeThreshold = 0.001;
		double mDLogLikehoodThreshold = 100;
		int mComponentCount = 4;
		bool mKMeansInitialized = false;
		bool mGMMInitialized = false;
		bool mRandomSeed = false;
		int mSegType = 0;
		const char* mInputModel = nullptr;
		const char* mOutputModel = nullptr;
		double mBeta = 12.0;
		int mWindSize = 5;
		ColorMap* mColorMap;
	};

	GMM_API struct SegOutput {
		int OutWidth;
		int OutHeight;
		uchar* OutPixels;
		int IterationCount = 0;
		double* LogLikehoodSummary = nullptr;
		char* ModelString = nullptr;
	};

	GMM_API void Segmentation(const SegArg* Arg, SegOutput* Output);
	GMM_API void FreeOutput(SegOutput* Output);

	//////////////// IMPLEMENTATION ////////////////
	GMM_API void Segmentation(const SegArg* Arg, SegOutput* Output) {
		std::vector<std::vector<uchar>>ColorMap{};
		auto Map = Arg->mColorMap;
		for (int i = 0; i < Map->Count; i++) {
			ColorMap.push_back({ Map->Colors[i].R, Map->Colors[i].G, Map->Colors[i].B });
		}
		GMM::SegArg GMMArg = GMM::SegArg()
			.InputImageName(Arg->InputImageName)
			.Beta(Arg->mBeta)
			.ColorMap(ColorMap)
			.ComponentCount(Arg->mComponentCount)
			.DCoeThreshold(Arg->mDCoeThreshold)
			.DExpThreshold(Arg->mDExpThreshold)
			.DLogLikehoodThreshold(Arg->mDLogLikehoodThreshold)
			.DVarThreshold(Arg->mDVarThreshold)
			.InputModel(Arg->mInputModel)
			.KMeansInitialized(Arg->mKMeansInitialized)
			.GMMInitialized(Arg->mGMMInitialized)
			.MaxIterationCount(Arg->mMaxIterationCount)
			.OutputModel(Arg->mOutputModel)
			.RandomSeed(Arg->mRandomSeed)
			.SegType(GMM::SegArg::ESegType(Arg->mSegType))
			.WindSize(Arg->mWindSize);

		printf("Arg: \n%s\n", GMMArg.ToString().c_str());

		GMM::SegOutput GMMOutput;
		GMM::Segmentation(GMMArg, GMMOutput);

		Output->OutWidth = GMMOutput.SegmentedImage.cols;
		Output->OutHeight = GMMOutput.SegmentedImage.rows;
		Output->OutPixels = (uchar*)std::malloc(sizeof(uchar) * Output->OutWidth * Output->OutHeight * 3);
		for (int i = 0; i < Output->OutHeight; i++) {
			for (int j = 0; j < Output->OutWidth; j++) {
				int Offset = (i * Output->OutWidth + j) * 3;
				auto Pixel = GMMOutput.SegmentedImage.at<cv::Vec3b>(i, j);
				Output->OutPixels[Offset] = Pixel[0];
				Output->OutPixels[Offset + 1] = Pixel[1];
				Output->OutPixels[Offset + 2] = Pixel[2];
			}
		}


		//Output->ModelString = strdup(GMMOutput.ModelString.c_str());
		Output->ModelString = (char*)std::malloc(sizeof(char) * (GMMOutput.ModelString.size() + 1));
		std::memset(Output->ModelString, 0, sizeof(char) * (GMMOutput.ModelString.size() + 1));
		std::copy(GMMOutput.ModelString.begin(), GMMOutput.ModelString.end(), Output->ModelString);
		Output->IterationCount = GMMOutput.IterationCount;
		Output->LogLikehoodSummary = (double*)std::malloc(sizeof(double) * Output->IterationCount);
		std::copy(GMMOutput.LogLikehoodSummary.begin(), GMMOutput.LogLikehoodSummary.end(), Output->LogLikehoodSummary);
	}

	GMM_API void FreeOutput(SegOutput* Output) {
		if (Output) {
			if (Output->OutPixels) {
				//printf("free pixels\n");
				free(Output->OutPixels);
			}
			if (Output->ModelString) {
				//printf("free model string\n");
				free(Output->ModelString);
			}
			if (Output->LogLikehoodSummary) {
				//printf("free LogLikehoodSummary \n");
				free(Output->LogLikehoodSummary);
			}
		}
	}
}