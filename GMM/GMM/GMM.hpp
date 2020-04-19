//#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#define PI 3.141592653589793238463
#define E  2.718281828459045235360
#define SQUARE(X) ((X) * (X))
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
}

namespace GMM {
	struct SegArg {
		enum class ESegType { GMMGray, GMMColor, KMeansGray, KMeansColor, MGMMGray, MGMMColor };

		int mMaxIterationCount = 20;
		double mDVarThreshold = 0.001;
		double mDExpThreshold = 0.001;
		double mDCoeThreshold = 0.001;
		double mDLogLikehoodThreshold = 100;
		int mComponentCount = 4;
		bool mKMeansInitialized = false;
		bool mRandomSeed = false;
		ESegType mSegType = ESegType::GMMGray;
		const char* mInputModel = nullptr;
		const char* mOutputModel = nullptr;

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

		bool IsGray() const {
			return this->mSegType == ESegType::GMMGray || this->mSegType == ESegType::MGMMGray || this->mSegType == ESegType::KMeansGray;
		}

		bool IsEM() const {
			return mSegType != ESegType::KMeansGray && mSegType != ESegType::KMeansColor;
		}
	};

	struct GaussianDistribution1D {
		double Expectation = 0;
		double Variance = 0.1;
		double Evaluate(double X) const {
			return (1.0 / (std::sqrt(2.0 * PI * Variance))) * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
		}
	};

	struct GaussianDistribution3D {
		cv::Vec3d Expectation = { 0, 0, 0 };
		cv::Mat VarianceMat = (cv::Mat_<double>(3, 3) << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1);
		double Cache1 = 0;
		cv::Mat Cache2;
		bool UseCache = true;
		double Evaluate(const cv::Vec3d& X) const {
			auto TmpMat = cv::Mat(X - Expectation).t();
			if (!UseCache) {
				return 1.0 / std::sqrt(std::pow(2.0 * PI, 3) * cv::determinant(VarianceMat))
					* std::pow(E, -0.5 * (TmpMat * VarianceMat.inv()).dot(TmpMat));
			}
			else {
				return Cache1
					* std::pow(E, -0.5 * (TmpMat * Cache2).dot(TmpMat));
			}
		}
		void UpdateCache() {
			Cache1 = 1.0 / std::sqrt(std::pow(2.0 * PI, 3) * cv::determinant(VarianceMat));
			Cache2 = VarianceMat.inv();
		}
	};

	struct EMContext {
		int IterationCount = 0;
		std::vector<double>DExp;
		std::vector<double>DVar;
		std::vector<double>DCoeff;
		double DLogLikehood;
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
		std::vector<cv::Mat> PostProbility;
		MixtureModel(int K) :Context(EMContext(K)),
			PostProbility(std::vector<cv::Mat>(K)) {}
		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true) {
			for (int i = 0; i < PostProbility.size(); i++) {
				PostProbility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
			}
		}
		virtual void EStep(const cv::Mat& InImage) = 0;
		virtual void MStep(const cv::Mat& InImage) = 0;
		virtual std::string TypeString() const = 0;
		virtual std::string ToString() const = 0;
		virtual void Save(const char* Filename) = 0;
		virtual void Load(const char* Filename) = 0;
	};

	static void KMeansGray(const cv::Mat& InImage, const int K, std::vector<double>& OutMeans) {
		printf("-------- K-Means --------\n");
		OutMeans.resize(K);
		// 随机初始化
		for (int i = 0; i < K; i++) {
			int RandRow = std::rand() % InImage.rows;
			int RandCol = std::rand() % InImage.cols;
			OutMeans[i] = (InImage.at<double>(RandRow, RandCol));
		}

		struct Cluster {
			double Sum = 0.0;
			int Count = 0;
			void Add(double Sample) {
				Sum += Sample;
				Count++;
			}
			double Center() {
				assert(Count > 0);
				return Sum / Count;
			}
		};
		std::vector<Cluster>Clusteres(K);
		int N = InImage.rows * InImage.cols;
		bool HasMeansUpdated = false;
		// K-Means
		do {
			std::fill(Clusteres.begin(), Clusteres.end(), Cluster{ 0, 0 });
			// a) 将样本划入簇
			for (int i = 0; i < N; i++) {
				double Sample = InImage.at<double>(i);
				double MinError = 0, MinIndex = 0;
				for (int j = 0; j < K; j++) {
					auto Error = std::sqrt(SQUARE(Sample - OutMeans[j]));
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
				double Diff = std::abs(OutMeans[i] - Clusteres[i].Center());
				printf("Diff #%d: %.6f\n", i, Diff);
				if (Diff > Threshold) {
					OutMeans[i] = Clusteres[i].Center();
					HasMeansUpdated = true;
				}
			}
		} while (HasMeansUpdated);
	}

	static void KMeansColor(const cv::Mat& InImage, const int K, std::vector<cv::Vec3d>& OutMeans) {
		printf("-------- K-Means --------\n");
		OutMeans.resize(K);
		// 随机初始化
		for (int i = 0; i < K; i++) {
			int RandRow = std::rand() % InImage.rows;
			int RandCol = std::rand() % InImage.cols;
			OutMeans[i] = (InImage.at<cv::Vec3d>(RandRow, RandCol));
		}

		struct Cluster {
			cv::Vec3d Sum = { 0, 0, 0 };
			int Count = 0;
			void Add(cv::Vec3d Sample) {
				Sum += Sample;
				Count++;
			}
			cv::Vec3d Center() {
				assert(Count > 0);
				return Sum / Count;
			}
		};
		std::vector<Cluster>Clusteres(K);
		int N = InImage.rows * InImage.cols;
		bool HasMeansUpdated = false;
		static auto Vec3dHypot = [](const cv::Vec3d& In) -> double {
			return std::sqrt(
				In[0] * In[0] +
				In[1] * In[1] +
				In[2] * In[2]
			);
		};
		static auto Vec3dAbs = [](const cv::Vec3d& In)->double {
			return (std::abs(In[0]) + std::abs(In[1]) + std::abs(In[2])) / 3.0;
		};
		// K-Means
		do {
			std::fill(Clusteres.begin(), Clusteres.end(), Cluster{ 0, 0 });
			// a) 将样本划入簇
			for (int i = 0; i < N; i++) {
				cv::Vec3d Sample = InImage.at<cv::Vec3d>(i);

				double MinError = 0, MinIndex = 0;
				for (int j = 0; j < K; j++) {
					auto Error = Vec3dHypot(Sample - OutMeans[j]);
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
				double Diff = Vec3dAbs(OutMeans[i] - Clusteres[i].Center());
				printf("Diff #%d: %.6f\n", i, Diff);
				if (Diff > Threshold) {
					OutMeans[i] = Clusteres[i].Center();
					HasMeansUpdated = true;
				}
			}
		} while (HasMeansUpdated);
	}

	struct GaussianMixtureModel1D : MixtureModel
	{
		int K;
		std::vector<GaussianDistribution1D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		GaussianMixtureModel1D(const int K) :
			MixtureModel(K),
			K(K),
			GaussianDistributions(std::vector<GaussianDistribution1D>(K)),
			MixtureCoefficients(std::vector<double>(K)) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true) override {
			printf("Initialize....\n");

			if (UseKMeansInitialize) {
				std::vector<double>Means;
				KMeansGray(InImage, K, Means);
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).Expectation = Means[i];
				}
			}
			else {
				for (int i = 0; i < Count(); i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GetGaussianDistribution(i).Expectation = InImage.at<double>(RandRow, RandCol);
				}
			}

			std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

			MixtureModel::Initialize(InImage, UseKMeansInitialize);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			for (int i = 0; i < Count(); i++) {
				for (int row = 0; row < PostProbility[i].rows; row++) {
					for (int col = 0; col < PostProbility[i].cols; col++) {
						double Up = GetMixtureCoefficient(i) * GetGaussianDistribution(i).Evaluate(InImage.at<double>(row, col));
						double Sum = 0;
						for (int j = 0; j < Count(); j++) {
							Sum += GetMixtureCoefficient(j) * GetGaussianDistribution(j).Evaluate(InImage.at<double>(row, col));
						}
						PostProbility[i].at<double>(row, col) = Up / Sum;
					}
				}
			}
		}

		virtual void MStep(const cv::Mat& InImage) override {
			for (int i = 0; i < Count(); i++) {
				double SumProbility = 0.0;
				double SumExpectation = 0.0;
				for (int row = 0; row < PostProbility[i].rows; row++) {
					for (int col = 0; col < PostProbility[i].cols; col++) {
						SumProbility += PostProbility[i].at<double>(row, col);
						SumExpectation += PostProbility[i].at<double>(row, col) * InImage.at<double>(row, col);
					}
				}
				auto N = PostProbility[i].rows * PostProbility[i].cols;
				auto& OldCoeff = GetMixtureCoefficient(i);
				auto& OldGaussianDistrib = GetGaussianDistribution(i);
				auto NewCoeff = SumProbility / N;
				GaussianDistribution1D NewGaussianDistrib;
				NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

				double SumVariance = 0.0;
				for (int row = 0; row < PostProbility[i].rows; row++) {
					for (int col = 0; col < PostProbility[i].cols; col++) {
						SumVariance += PostProbility[i].at<double>(row, col) * std::pow(InImage.at<double>(row, col) - NewGaussianDistrib.Expectation, 2);
					}
				}
				NewGaussianDistrib.Variance = SumVariance / SumProbility;

				Context.DCoeff[i] = std::abs(OldCoeff - NewCoeff);
				Context.DExp[i] = std::abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
				Context.DVar[i] = std::abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);
				Context.DLogLikehood = 0;

				OldCoeff = NewCoeff;
				OldGaussianDistrib = NewGaussianDistrib;
			}
		}

		virtual std::string TypeString() const override {
			return "GaussianMixtureModel 1D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Coe: %.6f\tExp: %.6f\tVar: %.6f\n";
			char Buffer[256];
			for (int i = 0; i < Count(); i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, MixtureCoefficients[i], GaussianDistributions[i].Expectation, GaussianDistributions[i].Variance);
				Stream << Buffer;
			}
			Stream << "\n";
			return Stream.str();
		}

		virtual void Save(const char* Filename) override {
			FILE* File = fopen(Filename, "wb");
			auto DistSize = GaussianDistributions.size();
			fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
			fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution1D), DistSize, File);
			auto MixSize = MixtureCoefficients.size();
			fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
			fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
			fclose(File);
		}

		virtual void Load(const char* Filename) override {
			FILE* File = fopen(Filename, "rb");
			size_t DistSize = 0, MixSize = 0;

			fread(&DistSize, sizeof(size_t), 1, File);
			GaussianDistributions.resize(DistSize);
			fread(&GaussianDistributions[0], sizeof(GaussianDistribution1D), DistSize, File);

			fread(&MixSize, sizeof(size_t), 1, File);
			MixtureCoefficients.resize(MixSize);
			fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

			fclose(File);
		}

		int Count() const {
			return GaussianDistributions.size();
		}

		double GetMixtureCoefficient(int Index) const {
			assert(0 <= Index && Index < Count());
			return MixtureCoefficients[Index];
		}

		double& GetMixtureCoefficient(int Index) {
			assert(0 <= Index && Index < Count());
			return MixtureCoefficients[Index];
		}

		GaussianDistribution1D& GetGaussianDistribution(int Index) {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		const GaussianDistribution1D& GetGaussianDistribution(int Index) const {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}


	};

	struct GaussianMixtureModel3D : MixtureModel
	{
		int K;
		std::vector<GaussianDistribution3D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		GaussianMixtureModel3D(const int K) :
			MixtureModel(K),
			K(K),
			GaussianDistributions(std::vector<GaussianDistribution3D>(K)),
			MixtureCoefficients(std::vector<double>(K)){
		}


		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true) override {

			printf("Initialize....\n");

			if (UseKMeansInitialize) {
				std::vector<cv::Vec3d>Means;
				KMeansColor(InImage, K, Means);
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).Expectation = Means[i];
				}
			}
			else {
				for (int i = 0; i < Count(); i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GetGaussianDistribution(i).Expectation = InImage.at<cv::Vec3d>(RandRow, RandCol);
				}
			}

			std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

			MixtureModel::Initialize(InImage, UseKMeansInitialize);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			// update cache
			for (int i = 0; i < Count(); i++) {
				GetGaussianDistribution(i).UpdateCache();
			}
			for (int i = 0; i < Count(); i++) {
				for (int row = 0; row < PostProbility[i].rows; row++) {
#pragma omp parallel for
					for (int col = 0; col < PostProbility[i].cols; col++) {
						double Up = GetMixtureCoefficient(i) * GetGaussianDistribution(i).Evaluate(InImage.at<cv::Vec3d>(row, col));
						double Sum = 0;
						for (int j = 0; j < Count(); j++) {
							Sum += GetMixtureCoefficient(j) * GetGaussianDistribution(j).Evaluate(InImage.at<cv::Vec3d>(row, col));
						}
						PostProbility[i].at<double>(row, col) = Up / Sum;
					}
				}
			}
		}

		virtual void MStep(const cv::Mat& InImage) override {
			for (int i = 0; i < Count(); i++) {
				double SumProbility = 0.0;
				cv::Vec3d SumExpectation = 0.0;
				for (int row = 0; row < PostProbility[i].rows; row++) {
					for (int col = 0; col < PostProbility[i].cols; col++) {
						SumProbility += PostProbility[i].at<double>(row, col);
						SumExpectation += PostProbility[i].at<double>(row, col) * InImage.at<cv::Vec3d>(row, col);
					}
				}
				auto N = PostProbility[i].rows * PostProbility[i].cols;
				auto& OldCoeff = GetMixtureCoefficient(i);
				auto& OldGaussianDistrib = GetGaussianDistribution(i);
				auto NewCoeff = SumProbility / N;
				auto NewGaussianDistribExpectation = SumExpectation / SumProbility;

				cv::Mat SumVariance = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
				for (int row = 0; row < PostProbility[i].rows; row++) {
					for (int col = 0; col < PostProbility[i].cols; col++) {
						cv::Vec3d Diff = InImage.at<cv::Vec3d>(row, col) - NewGaussianDistribExpectation;
						SumVariance += PostProbility[i].at<double>(row, col) * cv::Mat(Diff) * cv::Mat(Diff).t();
					}
				}
				auto NewGaussianDistribVariance = SumVariance / SumProbility;

				Context.DCoeff[i] = std::abs(OldCoeff - NewCoeff);
				Context.DExp[i] = Math::Vec3dAbs(OldGaussianDistrib.Expectation - NewGaussianDistribExpectation);
				Context.DVar[i] = Math::MatAbs(OldGaussianDistrib.VarianceMat - NewGaussianDistribVariance);
				Context.DLogLikehood = 0;

				OldCoeff = NewCoeff;
				OldGaussianDistrib.Expectation = NewGaussianDistribExpectation;
				OldGaussianDistrib.VarianceMat = NewGaussianDistribVariance;
			}
		}

		virtual std::string TypeString() const override {
			return "GaussianMixtureModel 3D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Coe: %.6f\tExp: %s\tVar: %s\n";
			char Buffer[256];
			for (int i = 0; i < Count(); i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, MixtureCoefficients[i],
					CV::ToString(GaussianDistributions[i].Expectation).c_str(),
					CV::ToString(GaussianDistributions[i].VarianceMat).c_str());
				Stream << Buffer;
			}
			Stream << "\n";
			return Stream.str();
		}

		int Count() const {
			return GaussianDistributions.size();
		}

		double GetMixtureCoefficient(int Index) const {
			assert(0 <= Index && Index < Count());
			return MixtureCoefficients[Index];
		}

		double& GetMixtureCoefficient(int Index) {
			assert(0 <= Index && Index < Count());
			return MixtureCoefficients[Index];
		}

		GaussianDistribution3D& GetGaussianDistribution(int Index) {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		const GaussianDistribution3D& GetGaussianDistribution(int Index) const {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		virtual void Save(const char* Filename) override {
			FILE* File = fopen(Filename, "wb");
			auto DistSize = GaussianDistributions.size();
			fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
			fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution3D), DistSize, File);
			auto MixSize = MixtureCoefficients.size();
			fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
			fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
			fclose(File);
		}

		virtual void Load(const char* Filename) override {
			FILE* File = fopen(Filename, "rb");
			size_t DistSize = 0, MixSize = 0;

			fread(&DistSize, sizeof(size_t), 1, File);
			GaussianDistributions.resize(DistSize);
			fread(&GaussianDistributions[0], sizeof(GaussianDistribution3D), DistSize, File);

			fread(&MixSize, sizeof(size_t), 1, File);
			MixtureCoefficients.resize(MixSize);
			fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

			fclose(File);
		}
	};

	struct ModifiedGaussianMixtureModel1D : MixtureModel
	{
		int K, N;
		std::vector<GaussianDistribution1D> GaussianDistributions;
		std::vector<double> MixtureCoefficients;
		std::vector<double>GCache;
		double LogLikehoodCache = 1;

		ModifiedGaussianMixtureModel1D(const int K, const int N) :
			MixtureModel(K),
			K(K), N(N),
			GaussianDistributions(std::vector<GaussianDistribution1D>(K)),
			MixtureCoefficients(std::vector<double>(K* size_t(N))),
			GCache(std::vector<double>(K* size_t(N))) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true) override {
			printf("Initialize....\n");

			if (UseKMeansInitialize) {
				std::vector<double>Means;
				KMeansGray(InImage, K, Means);
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).Expectation = Means[i];
				}
			}
			else {
				for (int i = 0; i < Count(); i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GetGaussianDistribution(i).Expectation = InImage.at<double>(RandRow, RandCol);
				}
			}

			std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

			MixtureModel::Initialize(InImage, UseKMeansInitialize);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			printf(">>>>>>>>>>>>>>>> E-Step....\n");
			for (int i = 0; i < Count(); i++) {
				printf("Estimate %d/%d\n", i + 1, Count());
				for (int NIndex = 0; NIndex < N; NIndex++) {
					double Up = GetMixtureCoefficient(i, NIndex) * GaussianDistributions[i].Evaluate(InImage.at<double>(NIndex));
					double Sum = 1e-9;
					for (int j = 0; j < Count(); j++) {
						Sum += GetMixtureCoefficient(j, NIndex) * GaussianDistributions[j].Evaluate(InImage.at<double>(NIndex));
					}
					PostProbility[i].at<double>(NIndex) = Up / Sum;
				}
			}
		}

		virtual void MStep(const cv::Mat& InImage) override {
			printf(">>>>>>>>>>>>>>>> M-Step....\n");
			// 计算上一次迭代的Gij
			{
				printf("Compute GCache....\n");
				for (int i = 0; i < InImage.rows; i++) {
					for (int j = 0; j < InImage.cols; j++) {
						for (int k = 0; k < Count(); k++) {
							int NIndex = i * InImage.cols + j;
							GCache[k * size_t(N) + NIndex] = G(k, NIndex, InImage.rows, InImage.cols);
						}
					}
				}
			}

			// 更新每个高斯分布
			{
				printf("Update Gaussian Parameters....\n");
				for (int i = 0; i < Count(); i++) {
					double SumProbility = 0.0;
					double SumExpectation = 0.0;
					for (int row = 0; row < PostProbility[i].rows; row++) {
						for (int col = 0; col < PostProbility[i].cols; col++) {
							SumProbility += PostProbility[i].at<double>(row, col);
							SumExpectation += PostProbility[i].at<double>(row, col) * InImage.at<double>(row, col);
						}
					}

					auto& OldGaussianDistrib = GetGaussianDistribution(i);
					GaussianDistribution1D NewGaussianDistrib;

					// 期望
					NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

					// 方差
					double SumVariance = 0.0;
					for (int row = 0; row < PostProbility[i].rows; row++) {
						for (int col = 0; col < PostProbility[i].cols; col++) {
							SumVariance += PostProbility[i].at<double>(row, col) * SQUARE(InImage.at<double>(row, col) - NewGaussianDistrib.Expectation);
						}
					}
					NewGaussianDistrib.Variance = SumVariance / SumProbility;

					Context.DExp[i] = std::abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
					Context.DVar[i] = std::abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);
					Context.DCoeff[i] = 0;

					OldGaussianDistrib = NewGaussianDistrib;
				}
			}

			// 更新系数
			{
				printf("Update Coefficient....\n");
				for (int i = 0; i < InImage.rows; i++) {
					for (int j = 0; j < InImage.cols; j++) {
						int NIndex = i * InImage.cols + j;
						double SumZG = 0.0;
						for (int k = 0; k < K; k++) {
							SumZG += (PostProbility[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols));
						}

						for (int k = 0; k < K; k++) {
							auto ZG = (PostProbility[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols));
							GetMixtureCoefficient(k, NIndex) = ZG / SumZG;
						}
					}
				}
			}

			// 计算似然函数
			{
				printf("Compute LogLikehood: ");
				// first term
				double FirstTerm = 0;
				for (int i = 0; i < N; i++) {
					double SumTemp = 1e-9;
					for (int k = 0; k < K; k++) {
						SumTemp += GetMixtureCoefficient(k, i)
							* GaussianDistributions[k].Evaluate(InImage.at<double>(i));
					}
					FirstTerm += std::log(SumTemp);
				}
				// second term
				double SecondTerm = 0;
				for (int i = 0; i < N; i++) {
					double SumTemp = 1e-9;
					for (int k = 0; k < K; k++) {
						SumTemp += GCache[k * size_t(N) + i]
							* std::log(GetMixtureCoefficient(k, i));
					}
					SecondTerm += SumTemp;
				}

				auto LogLikehood = FirstTerm + SecondTerm;
				if (LogLikehoodCache <= 0) {
					Context.DLogLikehood = std::abs(LogLikehood - LogLikehoodCache);
				}
				LogLikehoodCache = LogLikehood;
				printf("%f\n", LogLikehood);
			}
		}

		virtual std::string TypeString() const override {
			return "ModifiedGaussianMixtureModel 1D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Exp: %.6f\tVar: %.6f\n";
			char Buffer[256];
			for (int i = 0; i < Count(); i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i, GaussianDistributions[i].Expectation, GaussianDistributions[i].Variance);
				Stream << Buffer;
			}
			return Stream.str();
		}

		virtual void Save(const char* Filename) override {
			FILE* File = fopen(Filename, "wb");
			auto DistSize = GaussianDistributions.size();
			fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
			fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution1D), DistSize, File);
			auto MixSize = MixtureCoefficients.size();
			fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
			fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
			fclose(File);
		}

		virtual void Load(const char* Filename) override {
			FILE* File = fopen(Filename, "rb");
			size_t DistSize = 0, MixSize = 0;

			fread(&DistSize, sizeof(size_t), 1, File);
			GaussianDistributions.resize(DistSize);
			fread(&GaussianDistributions[0], sizeof(GaussianDistribution1D), DistSize, File);

			fread(&MixSize, sizeof(size_t), 1, File);
			MixtureCoefficients.resize(MixSize);
			fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

			fclose(File);
		}

		int Count() const {
			return GaussianDistributions.size();
		}

		double GetMixtureCoefficient(int KIndex, int NIndex) const {
			assert(0 <= KIndex && KIndex < K && 0 <= NIndex && NIndex < N);
			return MixtureCoefficients[KIndex * size_t(N) + NIndex];
		}

		double& GetMixtureCoefficient(int KIndex, int NIndex) {
			assert(0 <= KIndex && KIndex < K && 0 <= NIndex && NIndex < N);
			return MixtureCoefficients[KIndex * size_t(N) + NIndex];
		}

		GaussianDistribution1D& GetGaussianDistribution(int Index) {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		const GaussianDistribution1D& GetGaussianDistribution(int Index) const {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		double G(int KIndex, int NIndex, int Rows, int Cols) {
			const double Beta = 12;
			const int WindSize = 5;
			const int Ni = WindSize * WindSize;
			double Sum = 0.0;
			int Row = NIndex / Cols;
			int Col = NIndex % Cols;
			for (int i = -WindSize / 2; i <= WindSize / 2; i++) {
				for (int j = -WindSize / 2; j <= WindSize / 2; j++) {
					int R = Row + i, C = Col + j;
					double Z = 0, M = 0;
					if (0 <= R && R < Rows && 0 <= C && C < Cols) {
						Z = PostProbility[KIndex].at<double>(R, C);
						M = GetMixtureCoefficient(KIndex, R * Cols + C);
					}
					Sum += (Z + M);
				}
			}
			return std::pow(E, (Beta / (2.0 * Ni)) * Sum);
		}
	};

	struct ModifiedGaussianMixtureModel3D : MixtureModel
	{
		int K, N;
		std::vector<double> MixtureCoefficients;
		std::vector<GaussianDistribution3D> GaussianDistributions;
		std::vector<double>GCache;
		double LogLikehoodCache = 1;

		ModifiedGaussianMixtureModel3D(const int K, const int N) :
			MixtureModel(K),
			K(K), N(N),
			GaussianDistributions(std::vector<GaussianDistribution3D>(K)),
			MixtureCoefficients(std::vector<double>(K* size_t(N))),
			GCache(std::vector<double>(K* size_t(N))) {
		}

		virtual void Initialize(const cv::Mat& InImage, bool UseKMeansInitialize = true) override {
			printf("Initialize....\n");

			if (UseKMeansInitialize) {
				std::vector<cv::Vec3d>Means;
				KMeansColor(InImage, K, Means);
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).Expectation = Means[i];
				}
			}
			else {
				for (int i = 0; i < Count(); i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GetGaussianDistribution(i).Expectation = InImage.at<cv::Vec3d>(RandRow, RandCol);
				}
			}

			std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

			MixtureModel::Initialize(InImage, UseKMeansInitialize);
		}

		virtual void EStep(const cv::Mat& InImage) override {
			this->UpdateCache();
			printf(">>>>>>>>>>>>>>>> E-Step....\n");
			for (int i = 0; i < Count(); i++) {
				printf("Estimate %d/%d\n", i + 1, Count());
#pragma omp parallel for
				for (int NIndex = 0; NIndex < N; NIndex++) {
					double Up = GetMixtureCoefficient(i, NIndex) * GaussianDistributions[i].Evaluate(InImage.at<cv::Vec3d>(NIndex));
					double Sum = 1e-9;
					for (int j = 0; j < Count(); j++) {
						Sum += GetMixtureCoefficient(j, NIndex) * GaussianDistributions[j].Evaluate(InImage.at<cv::Vec3d>(NIndex));
					}
					PostProbility[i].at<double>(NIndex) = Up / Sum;
				}
			}
		}

		virtual void MStep(const cv::Mat& InImage) override {
			printf(">>>>>>>>>>>>>>>> M-Step....\n");
			// 计算上一次迭代的Gij
			{
				printf("Compute GCache....\n");
#pragma omp parallel for
				for (int i = 0; i < InImage.rows; i++) {
					for (int j = 0; j < InImage.cols; j++) {
						for (int k = 0; k < Count(); k++) {
							int NIndex = i * InImage.cols + j;
							GCache[k * size_t(N) + NIndex] = G(k, NIndex, InImage.rows, InImage.cols);
						}
					}
				}
			}

			// 更新每个高斯分布
			{
				printf("Update Gaussian Parameters....\n");
				for (int i = 0; i < Count(); i++) {
					double SumProbility = 0.0;
					cv::Vec3d SumExpectation = 0.0;
					for (int row = 0; row < PostProbility[i].rows; row++) {
						for (int col = 0; col < PostProbility[i].cols; col++) {
							SumProbility += PostProbility[i].at<double>(row, col);
							SumExpectation += PostProbility[i].at<double>(row, col) * InImage.at<cv::Vec3d>(row, col);
						}
					}

					auto& OldGaussianDistrib = GetGaussianDistribution(i);
					GaussianDistribution3D NewGaussianDistrib;

					// 期望
					NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

					// 方差
					cv::Mat SumVariance = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
					for (int row = 0; row < PostProbility[i].rows; row++) {
						for (int col = 0; col < PostProbility[i].cols; col++) {
							cv::Vec3d Diff = InImage.at<cv::Vec3d>(row, col) - NewGaussianDistrib.Expectation;
							SumVariance += PostProbility[i].at<double>(row, col) * cv::Mat(Diff) * cv::Mat(Diff).t();
						}
					}
					NewGaussianDistrib.VarianceMat = SumVariance / SumProbility;

					Context.DExp[i] = Math::Vec3dAbs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
					Context.DVar[i] = Math::MatAbs(OldGaussianDistrib.VarianceMat - NewGaussianDistrib.VarianceMat);
					Context.DCoeff[i] = 0;

					OldGaussianDistrib = NewGaussianDistrib;
				}
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
							SumZG += (PostProbility[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols));
						}

						for (int k = 0; k < K; k++) {
							auto ZG = (PostProbility[k].at<double>(i, j)
								+ G(k, NIndex, InImage.rows, InImage.cols));
							GetMixtureCoefficient(k, NIndex) = ZG / SumZG;
						}
					}
				}
			}

			// 计算似然函数
			{
				printf("Compute LogLikehood: ");
				this->UpdateCache();
				// first term
				double FirstTerm = 0;
#pragma omp parallel for reduction (+:FirstTerm)
				for (int i = 0; i < N; i++) {
					double SumTemp = 1e-9;
					for (int k = 0; k < K; k++) {
						SumTemp += GetMixtureCoefficient(k, i)
							* GaussianDistributions[k].Evaluate(InImage.at<cv::Vec3d>(i));
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
							* std::log(GetMixtureCoefficient(k, i));
					}
					SecondTerm += SumTemp;
				}

				auto LogLikehood = FirstTerm + SecondTerm;
				if (LogLikehoodCache <= 0) {
					Context.DLogLikehood = std::abs(LogLikehood - LogLikehoodCache);
				}
				LogLikehoodCache = LogLikehood;
				printf("%f\n", LogLikehood);
			}
		}

		virtual std::string TypeString() const override {
			return "ModifiedGaussianMixtureModel 3D";
		}

		virtual std::string ToString() const override {
			std::stringstream Stream;
			const char* Format = "#%d Exp: %s\tVar: %s\n";
			char Buffer[256];
			for (int i = 0; i < Count(); i++) {
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, i,
					CV::ToString(GaussianDistributions[i].Expectation).c_str(),
					CV::ToString(GaussianDistributions[i].VarianceMat).c_str());
				Stream << Buffer;
			}
			return Stream.str();
		}

		virtual void Save(const char* Filename) override {
			FILE* File = fopen(Filename, "wb");
			auto DistSize = GaussianDistributions.size();
			fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
			fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution3D), DistSize, File);
			auto MixSize = MixtureCoefficients.size();
			fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
			fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
			fclose(File);
		}

		virtual void Load(const char* Filename) override {
			FILE* File = fopen(Filename, "rb");
			size_t DistSize = 0, MixSize = 0;

			fread(&DistSize, sizeof(size_t), 1, File);
			GaussianDistributions.resize(DistSize);
			fread(&GaussianDistributions[0], sizeof(GaussianDistribution3D), DistSize, File);

			fread(&MixSize, sizeof(size_t), 1, File);
			MixtureCoefficients.resize(MixSize);
			fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

			fclose(File);
		}

		double G(int KIndex, int NIndex, int Rows, int Cols) {
			const double Beta = 12;
			const int WindSize = 5;
			const int Ni = WindSize * WindSize;
			double Sum = 0.0;
			int Row = NIndex / Cols;
			int Col = NIndex % Cols;
			for (int i = -WindSize / 2; i <= WindSize / 2; i++) {
				for (int j = -WindSize / 2; j <= WindSize / 2; j++) {
					int R = Row + i, C = Col + j;
					double Z = 0, M = 0;
					if (0 <= R && R < Rows && 0 <= C && C < Cols) {
						Z = PostProbility[KIndex].at<double>(R, C);
						M = GetMixtureCoefficient(KIndex, R * Cols + C);
					}
					Sum += (Z + M);
				}
			}
			return std::pow(E, (Beta / (2.0 * Ni)) * Sum);
		}

		int Count() const {
			return GaussianDistributions.size();
		}

		double GetMixtureCoefficient(int KIndex, int NIndex) const {
			assert(0 <= KIndex && KIndex < K && 0 <= NIndex && NIndex < N);
			return MixtureCoefficients[KIndex * size_t(N) + NIndex];
		}

		double& GetMixtureCoefficient(int KIndex, int NIndex) {
			assert(0 <= KIndex && KIndex < K && 0 <= NIndex && NIndex < N);
			return MixtureCoefficients[KIndex * size_t(N) + NIndex];
		}

		GaussianDistribution3D& GetGaussianDistribution(int Index) {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		const GaussianDistribution3D& GetGaussianDistribution(int Index) const {
			assert(0 <= Index && Index < Count());
			return GaussianDistributions[Index];
		}

		void UpdateCache() {
			// update cache
			for (int i = 0; i < Count(); i++) {
				GetGaussianDistribution(i).UpdateCache();
			}
		}
	};

	struct EMAlgorithm {
		static void Train(MixtureModel& Model, const cv::Mat& InImage, bool UseKMeansInitialize = true) {
			Model.Initialize(InImage, UseKMeansInitialize);
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

	static void KMeansSegmentationGray(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		std::vector<double> Means;
		KMeansGray(InImage, Arg.mComponentCount, Means);
		printf("-------- Segmentation --------\n");
		printf("Means: [");
		for (int i = 0; i < Means.size(); i++) {
			printf("%f", Means[i]);
			if (i != Means.size() - 1)printf(", ");
			else printf("]\n");
		}
		int N = InImage.rows * InImage.cols;
		double Step = 1.0 / (double(K) - 1);
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MinError = 0, MinIndex = 0;
			for (int j = 0; j < K; j++) {
				auto Error = std::sqrt(SQUARE(InImage.at<double>(Position) - Means[j]));
				if (j == 0 || Error < MinError) {
					MinError = Error;
					MinIndex = j;
				}
			}
			Pixel = Step * MinIndex;
		});
	}

	static void KMeansSegmentationColor(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		std::vector<cv::Vec3d> Means;
		KMeansColor(InImage, Arg.mComponentCount, Means);
		printf("-------- Segmentation --------\n");
		printf("Means: [");
		for (int i = 0; i < Means.size(); i++) {
			printf("(%f, %f, %f)", Means[i][0], Means[i][1], Means[i][2]);
			if (i != Means.size() - 1)printf(", ");
			else printf("]\n");
		}
		int N = InImage.rows * InImage.cols;
		double Step = 1.0 / (double(K) - 1);
		static auto Vec3dHypot = [](const cv::Vec3d& In) -> double {
			return std::sqrt(
				In[0] * In[0] +
				In[1] * In[1] +
				In[2] * In[2]
			);
		};
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MinError = 0, MinIndex = 0;
			for (int j = 0; j < K; j++) {
				auto Error = Vec3dHypot(InImage.at<cv::Vec3d>(Position) - Means[j]);
				if (j == 0 || Error < MinError) {
					MinError = Error;
					MinIndex = j;
				}
			}
			Pixel = Step * MinIndex;
		});
	}

	static void Segmentation(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
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
				Model = new ModifiedGaussianMixtureModel3D(Arg.mComponentCount, InImage.rows * InImage.cols);
				break;
			case SegArg::ESegType::MGMMGray:
				Model = new ModifiedGaussianMixtureModel1D(Arg.mComponentCount, InImage.rows * InImage.cols);
				break;
			default:
				break;
			}
			
			if (Model) {
				EMAlgorithm::Train(*Model, InImage, Arg.mKMeansInitialized);
				////////////////// Segmenation ////////////////
				Model->EStep(InImage);
				double Step = 1.0 / (double(Arg.mComponentCount) - 1);
				OutImage.forEach<double>(
					[&](double& Pixel, const int* Position) {
					double MaxProbility = 0.0;
					int MaxI = 0;
					auto& Probility = Model->PostProbility;
					for (int i = 0; i < Probility.size(); i++) {
						if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
							MaxI = i;
							MaxProbility = Probility[i].at<double>(Position);
						}
					}
					Pixel = Step * MaxI;
				}
				);
				//delete Model;
			}
		}
		else {
			switch (Arg.mSegType)
			{
			case SegArg::ESegType::KMeansColor:
				KMeansSegmentationColor(InImage, OutImage, Arg);
				break;
			case SegArg::ESegType::KMeansGray:
				KMeansSegmentationGray(InImage, OutImage, Arg);
				break;
			default:
				break;
			}
		}
	}

	static void TestSegmentation(const char* InputImageName, const SegArg& Arg) {
		bool OK = true;
		cv::Mat InputImage, SegmentedImg;

		if (Arg.IsGray()) {
			OK = CV::ReadImage(InputImageName, InputImage, cv::IMREAD_GRAYSCALE);
			InputImage.convertTo(InputImage, CV_64FC1, 1.0 / 255.0);
		}
		else {
			OK = CV::ReadImage(InputImageName, InputImage, cv::IMREAD_COLOR);
			InputImage.convertTo(InputImage, CV_64FC3, 1.0 / 255.0);
		}

		SegmentedImg = cv::Mat(InputImage.rows, InputImage.cols, CV_64FC1);
		Segmentation(InputImage, SegmentedImg, Arg);

		if (!OK) {
			printf("read image fail\n");
		}
		else {
			std::stringstream Stream;
			static int Count = 0;
			Stream << "Origin " << Count;
			CV::DisplayImage(InputImage, Stream.str().c_str());
			Stream.clear();
			Stream = std::stringstream();
			Stream << "Segmented" << Count;
			CV::DisplayImage(SegmentedImg, Stream.str().c_str());
			Count++;
		}
	}

	static void Main() {
		std::vector<std::pair<const char*, int>>TestData{
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\blog\\002\\0.png", 4						},
			{"D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f02970.png", 4	},
			{"D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f00300.png", 4	},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\grid.PNG", 4						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\horse2.PNG", 2						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\Lenna.jpg", 4						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\elephant.PNG", 2					},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\leaf.jpg", 2						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\snow.PNG", 3						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\cow.PNG", 4						},
			{"C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\horse1.PNG", 2						},
		};

		auto Arg = SegArg()
			.RandomSeed(true)
			.InputModel(nullptr)
			.DCoeThreshold(0.001)
			.DExpThreshold(0.001)
			.DVarThreshold(0.001)
			.OutputModel(nullptr)
			.MaxIterationCount(20)
			.KMeansInitialized(true)
			.DLogLikehoodThreshold(100)
			.ComponentCount(TestData.back().second);

		TestSegmentation(
			TestData.back().first,
			Arg
				.SegType(SegArg::ESegType::KMeansColor)
				.SegType(SegArg::ESegType::GMMGray)
				.SegType(SegArg::ESegType::GMMColor)
				.SegType(SegArg::ESegType::MGMMColor)
				.SegType(SegArg::ESegType::MGMMGray)
		);

		//TestSegmentation(
		//	TestData.back().first,
		//	Arg.SegType(SegArg::ESegType::KMeansGray)
		//);

		CV::Wait();
	}
}
