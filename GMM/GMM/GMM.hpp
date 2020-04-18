#pragma once
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

	static void Wait(int Delay=0) {
		cv::waitKey(Delay);
	}
}

namespace GMM {
	struct SegArg {
		enum class ESegType{GMMGray, GMMColor, KMeansGray, KMeansColor, MGMMGray, MGMMColor};
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
			std::fill(Clusteres.begin(), Clusteres.end(), Cluster{0, 0});
			// a) 将样本划入簇
			for (int i = 0; i < N; i++) {
				double Sample = InImage.at<double>(i);
				double MinError=0, MinIndex=0;
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
		static auto Vec3dHypot = [](const cv::Vec3d& In) -> double{
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

	static void KMeansSegmentationGray(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		std::vector<double> Means;
		KMeansGray(InImage, Arg.mComponentCount, Means);
		printf("-------- Segmentation --------\n");
		printf("Means: \[");
		for (int i = 0; i < Means.size(); i++) {
			printf("%f", Means[i]);
			if (i != Means.size() - 1)printf(", ");
			else printf("]\n");
		}
		int N = InImage.rows * InImage.cols;
		double Step = 1.0 / (double(K) - 1);
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
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
		printf("Means: \[");
		for (int i = 0; i < Means.size(); i++) {
			printf("(%f, %f, %f)", Means[i][0], Means[i][1], Means[i][2]);
			if (i != Means.size() - 1)printf(", ");
			else printf("]\n");
		}
		int N = InImage.rows * InImage.cols;
		double Step = 1.0 / (double(K) - 1);
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
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

	static void GMMSegmentationGray(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		assert(K > 1);
		//////////////// Type Def ////////////////
		struct GaussianDistribution {
			double Expectation = 0;
			double Variance = 0.1;
			double Evaluate(double X) const {
				return (1.0 / (std::sqrt(2.0 * PI * Variance))) * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
			}
		};

		struct GaussianMixtureModel {
			std::vector<GaussianDistribution> GaussianDistributions;
			std::vector<double> MixtureCoefficients;

			GaussianMixtureModel(const int K) :
				GaussianDistributions(std::vector<GaussianDistribution>(K)),
				MixtureCoefficients(std::vector<double>(K)) {
			}

			void Save(const char* Filename) {
				FILE* File = fopen(Filename, "wb");
				auto DistSize = GaussianDistributions.size();
				fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
				fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);
				auto MixSize = MixtureCoefficients.size();
				fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
				fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
				fclose(File);
			}

			void Load(const char* Filename) {
				FILE* File = fopen(Filename, "rb");
				size_t DistSize = 0, MixSize = 0;

				fread(&DistSize, sizeof(size_t), 1, File);
				GaussianDistributions.resize(DistSize);
				fread(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);

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

			GaussianDistribution& GetGaussianDistribution(int Index) {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			const GaussianDistribution& GetGaussianDistribution(int Index) const {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			std::string ToString() {
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
		};

		struct Context {
			int IterationCount = 0;
			std::vector<double>DCoeff;
			std::vector<double>DExp;
			std::vector<double>DVar;

			Context(int K) :
				DCoeff(std::vector<double>(K, DBL_MAX)),
				DExp(std::vector<double>(K, DBL_MAX)),
				DVar(std::vector<double>(K, DBL_MAX)) {
			}

			double MaxDCoeff() const {
				return MaxValue(DCoeff);
			}

			double MaxDExp() const {
				return MaxValue(DExp);
			}

			double MaxDVar() const {
				return MaxValue(DVar);
			}

			std::string ToString() {
				std::stringstream Stream;
				Stream << "Iteration\t: " << IterationCount << "\n"
					<< "Max DCoeff\t: " << MaxDCoeff() << "\n"
					<< "Max DExp\t: " << MaxDExp() << "\n"
					<< "Max DVar\t: " << MaxDVar() << "\n";
				return Stream.str();
			}
		private:
			double MaxValue(const std::vector<double>& Vec) const {
				return *std::max_element(Vec.begin(), Vec.end());
			}
		};
		//////////////// Method Def ////////////////
		// 初始化
		static auto Initialize = [](GaussianMixtureModel& Model, std::vector<cv::Mat>& Probility, const cv::Mat& InImage) {
			for (int i = 0; i < Model.Count(); i++) {
				Model.MixtureCoefficients[i] = 1.0 / Model.Count();
				int RandRow = std::rand() % InImage.rows;
				int RandCol = std::rand() % InImage.cols;
				Model.GetGaussianDistribution(i).Expectation = InImage.at<double>(RandRow, RandCol);
			}

			for (int i = 0; i < Probility.size(); i++) {
				Probility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
			}
		};

		// 是否满足停止条件
		static auto CheckCondition = [](const Context& Context) -> bool {
			return Context.IterationCount >= 20
				|| (Context.MaxDCoeff() <= 0.0001
					&& Context.MaxDExp() <= 0.0001
					&& Context.MaxDVar() <= 0.0001);
		};

		// 计算概率
		static auto ComputeProbability = [](const cv::Mat& InputImage, const GaussianMixtureModel& Model, std::vector<cv::Mat>& OutProbility) {
			for (int i = 0; i < Model.Count(); i++) {
				for (int row = 0; row < OutProbility[i].rows; row++) {
					for (int col = 0; col < OutProbility[i].cols; col++) {
						double Up = Model.GetMixtureCoefficient(i) * Model.GetGaussianDistribution(i).Evaluate(InputImage.at<double>(row, col));
						double Sum = 0;
						for (int j = 0; j < Model.Count(); j++) {
							Sum += Model.GetMixtureCoefficient(j) * Model.GetGaussianDistribution(j).Evaluate(InputImage.at<double>(row, col));
						}
						OutProbility[i].at<double>(row, col) = Up / Sum;
					}
				}
			}
		};

		// 更新参数
		static auto UpdateParameters = [](GaussianMixtureModel& Model, const cv::Mat& InImage, const std::vector<cv::Mat>& InProbility,
			std::vector<double>& DCoeff, std::vector<double>& DExp, std::vector<double>& DVar) {
			for (int i = 0; i < Model.Count(); i++) {
				double SumProbility = 0.0;
				double SumExpectation = 0.0;
				for (int row = 0; row < InProbility[i].rows; row++) {
					for (int col = 0; col < InProbility[i].cols; col++) {
						SumProbility += InProbility[i].at<double>(row, col);
						SumExpectation += InProbility[i].at<double>(row, col) * InImage.at<double>(row, col);
					}
				}
				auto N = InProbility[i].rows * InProbility[i].cols;
				auto& OldCoeff = Model.GetMixtureCoefficient(i);
				auto& OldGaussianDistrib = Model.GetGaussianDistribution(i);
				auto NewCoeff = SumProbility / N;
				GaussianDistribution NewGaussianDistrib;
				NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

				double SumVariance = 0.0;
				for (int row = 0; row < InProbility[i].rows; row++) {
					for (int col = 0; col < InProbility[i].cols; col++) {
						SumVariance += InProbility[i].at<double>(row, col) * std::pow(InImage.at<double>(row, col) - NewGaussianDistrib.Expectation, 2);
					}
				}
				NewGaussianDistrib.Variance = SumVariance / SumProbility;

				DCoeff[i] = std::abs(OldCoeff - NewCoeff);
				DExp[i] = std::abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
				DVar[i] = std::abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);

				OldCoeff = NewCoeff;
				OldGaussianDistrib = NewGaussianDistrib;
			}
		};

		//////////////// Implementation ////////////////
		// 初始化
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		GaussianMixtureModel Model(K);
		std::vector<cv::Mat> Probility(K);
		Context Context(K);
		Initialize(Model, Probility, InImage);

		if (Arg.mInputModel) {
			printf("Load model %s\n", Arg.mInputModel);
			Model.Load(Arg.mInputModel);
		}
		else {
			printf("-------- BEGIN --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s", Context.ToString().c_str());
			// 迭代求解
			while (!CheckCondition(Context)) {
				// E-Step
				ComputeProbability(InImage, Model, Probility);
				// M-Step
				UpdateParameters(Model, InImage, Probility,
					Context.DCoeff, Context.DExp, Context.DVar);
				// Update Context
				Context.IterationCount++;
			}
			printf("-------- End --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s", Context.ToString().c_str());
		}

		if (Arg.mOutputModel) {
			printf("Save model to %s\n", Arg.mOutputModel);
			Model.Save(Arg.mOutputModel);
		}

		//////////////// Segmenation ////////////////
		ComputeProbability(InImage, Model, Probility);
		double Step = 1.0 / double(K - 1);
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MaxProbility = 0.0;
			int MaxI = 0;
			for (int i = 0; i < Probility.size(); i++) {
				if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
					MaxI = i;
					MaxProbility = Probility[i].at<double>(Position);
				}
			}
			Pixel = Step * MaxI;
		}
		);
	}

	static void GMMSegmentationColor(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		assert(K > 1);
		//////////////// Type Def ////////////////
		struct GaussianDistribution {
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

		struct GaussianMixtureModel {
			std::vector<GaussianDistribution> GaussianDistributions;
			std::vector<double> MixtureCoefficients;

			GaussianMixtureModel(const int K) :
				GaussianDistributions(std::vector<GaussianDistribution>(K)),
				MixtureCoefficients(std::vector<double>(K)) {
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

			GaussianDistribution& GetGaussianDistribution(int Index) {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			const GaussianDistribution& GetGaussianDistribution(int Index) const {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			std::string ToString() {
				std::stringstream Stream;
				const char* Format = "#%d Coe: %.6f\tExp: %s\tVar: %s\n";
				char Buffer[256];
				for (int i = 0; i < Count(); i++) {
					memset(Buffer, 0, sizeof(Buffer));
					sprintf_s(Buffer, Format, i, MixtureCoefficients[i],
						ToString(GaussianDistributions[i].Expectation).c_str(),
						ToString(GaussianDistributions[i].VarianceMat).c_str());
					Stream << Buffer;
				}
				Stream << "\n";
				return Stream.str();
			}

		private:
			std::string ToString(const cv::Vec3d& Vec) {
				std::stringstream Stream;
				const char* Format = "[%.4f, %.4f, %.4f]";
				char Buffer[256];
				memset(Buffer, 0, sizeof(Buffer));
				sprintf_s(Buffer, Format, Vec[0], Vec[1], Vec[2]);
				Stream << Buffer;
				return Stream.str();
			}

			std::string ToString(const cv::Mat& Mat) {
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
		};

		struct Context {
			int IterationCount = 0;
			std::vector<double>DCoeff;
			std::vector<double>DExp;
			std::vector<double>DVar;

			Context(int K) :
				DCoeff(std::vector<double>(K, DBL_MAX)),
				DExp(std::vector<double>(K, DBL_MAX)),
				DVar(std::vector<double>(K, DBL_MAX)) {
			}

			double MaxDCoeff() const {
				return MaxValue(DCoeff);
			}

			double MaxDExp() const {
				return MaxValue(DExp);
			}

			double MaxDVar() const {
				return MaxValue(DVar);
			}

			std::string ToString() {
				std::stringstream Stream;
				Stream << "Iteration\t: " << IterationCount << "\n"
					<< "Max DCoeff\t: " << MaxDCoeff() << "\n"
					<< "Max DExp\t: " << MaxDExp() << "\n"
					<< "Max DVar\t: " << MaxDVar() << "\n";
				return Stream.str();
			}
		private:
			double MaxValue(const std::vector<double>& Vec) const {
				return *std::max_element(Vec.begin(), Vec.end());
			}
		};
		//////////////// Util ////////////////
		static auto Vec3dSquare = [](const cv::Vec3d& Input) -> cv::Vec3d {
			return cv::Vec3d(Input[0] * Input[0], Input[1] * Input[1], Input[2] * Input[2]);
		};
		static auto Vec3dAbs = [](const cv::Vec3d& Input) -> double {
			return (std::abs(Input[0]) + std::abs(Input[1]) + std::abs(Input[2])) / 3.0;
		};
		static auto MatAbs = [](const cv::Mat& Input)->double {
			double Sum = 0.0;
			for (int i = 0; i < Input.rows; i++) {
				for (int j = 0; j < Input.cols; j++) {
					Sum += Input.at<double>(i, j);
				}
			}
			return Sum / (Input.rows * Input.cols);
		};
		//////////////// Method Def ////////////////
		// 初始化
		static auto Initialize = [](GaussianMixtureModel& Model, std::vector<cv::Mat>& Probility, const cv::Mat& InImage) {
			for (int i = 0; i < Model.Count(); i++) {
				Model.MixtureCoefficients[i] = 1.0 / Model.Count();
				int RandRow = std::rand() % InImage.rows;
				int RandCol = std::rand() % InImage.cols;
				Model.GetGaussianDistribution(i).Expectation = InImage.at<cv::Vec3d>(RandRow, RandCol);
			}

			for (int i = 0; i < Probility.size(); i++) {
				Probility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
			}
		};

		// 是否满足停止条件
		static auto CheckCondition = [](const Context& Context) -> bool {
			return Context.IterationCount >= 20
				|| (Context.MaxDCoeff() <= 0.001
					&& Context.MaxDExp() <= 0.001
					&& Context.MaxDVar() <= 0.001);
		};

		// 计算概率
		static auto ComputeProbability = [](const cv::Mat& InputImage, GaussianMixtureModel& Model, std::vector<cv::Mat>& OutProbility) {
			// update cache
			for (int i = 0; i < Model.Count(); i++) {
				Model.GetGaussianDistribution(i).UpdateCache();
			}
			for (int i = 0; i < Model.Count(); i++) {
				for (int row = 0; row < OutProbility[i].rows; row++) {
					//auto Beg = std::clock();
#pragma omp parallel for
					for (int col = 0; col < OutProbility[i].cols; col++) {
						double Up = Model.GetMixtureCoefficient(i) * Model.GetGaussianDistribution(i).Evaluate(InputImage.at<cv::Vec3d>(row, col));
						double Sum = 0;
						for (int j = 0; j < Model.Count(); j++) {
							Sum += Model.GetMixtureCoefficient(j) * Model.GetGaussianDistribution(j).Evaluate(InputImage.at<cv::Vec3d>(row, col));
						}
						OutProbility[i].at<double>(row, col) = Up / Sum;
					}
					//auto End = std::clock();
					//std::cout << "use " << (End - Beg) << " ms" << std::endl;
				}
			}
		};

		// 更新参数
		static auto UpdateParameters = [](GaussianMixtureModel& Model, const cv::Mat& InImage, const std::vector<cv::Mat>& InProbility,
			std::vector<double>& DCoeff, std::vector<double>& DExp, std::vector<double>& DVar) {
			for (int i = 0; i < Model.Count(); i++) {
				double SumProbility = 0.0;
				cv::Vec3d SumExpectation = 0.0;
				for (int row = 0; row < InProbility[i].rows; row++) {
					for (int col = 0; col < InProbility[i].cols; col++) {
						SumProbility += InProbility[i].at<double>(row, col);
						SumExpectation += InProbility[i].at<double>(row, col) * InImage.at<cv::Vec3d>(row, col);
					}
				}
				auto N = InProbility[i].rows * InProbility[i].cols;
				auto& OldCoeff = Model.GetMixtureCoefficient(i);
				auto& OldGaussianDistrib = Model.GetGaussianDistribution(i);
				auto NewCoeff = SumProbility / N;
				auto NewGaussianDistribExpectation = SumExpectation / SumProbility;

				cv::Mat SumVariance = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
				for (int row = 0; row < InProbility[i].rows; row++) {
					for (int col = 0; col < InProbility[i].cols; col++) {
						cv::Vec3d Diff = InImage.at<cv::Vec3d>(row, col) - NewGaussianDistribExpectation;
						SumVariance += InProbility[i].at<double>(row, col) * cv::Mat(Diff) * cv::Mat(Diff).t();
					}
				}
				auto NewGaussianDistribVariance = SumVariance / SumProbility;

				DCoeff[i] = std::abs(OldCoeff - NewCoeff);
				DExp[i] = Vec3dAbs(OldGaussianDistrib.Expectation - NewGaussianDistribExpectation);
				DVar[i] = MatAbs(OldGaussianDistrib.VarianceMat - NewGaussianDistribVariance);

				OldCoeff = NewCoeff;
				OldGaussianDistrib.Expectation = NewGaussianDistribExpectation;
				OldGaussianDistrib.VarianceMat = NewGaussianDistribVariance;
			}
		};

		//////////////// Implementation ////////////////
		// 初始化
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		GaussianMixtureModel Model(K);
		std::vector<cv::Mat> Probility(K);
		Context Context(K);
		Initialize(Model, Probility, InImage);

		printf("-------- BEGIN --------\n");
		printf("Model: \n%s", Model.ToString().c_str());
		printf("Context: \n%s", Context.ToString().c_str());
		// 迭代求解
		while (!CheckCondition(Context)) {
			// E-Step
			ComputeProbability(InImage, Model, Probility);
			// M-Step
			UpdateParameters(Model, InImage, Probility, Context.DCoeff, Context.DExp, Context.DVar);
			// Update Context
			Context.IterationCount++;
			printf("======== Iter #%d ========\n", Context.IterationCount);
			printf("Model: \n%s", Model.ToString().c_str());
		}
		printf("-------- End --------\n");
		printf("Model: \n%s", Model.ToString().c_str());
		printf("Context: \n%s", Context.ToString().c_str());

		//////////////// Segmenation ////////////////
		ComputeProbability(InImage, Model, Probility);
		double Step = 1.0 / double(K - 1);
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MaxProbility = 0.0;
			int MaxI = 0;
			for (int i = 0; i < Probility.size(); i++) {
				if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
					MaxI = i;
					MaxProbility = Probility[i].at<double>(Position);
				}
			}
			Pixel = Step * MaxI;
		}
		);
	}

	static void ModifiedGMMSegmentationGray(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		assert(K > 1);
		//////////////// Type Def ////////////////
		struct GaussianDistribution {
			double Expectation = 0;
			double Variance = 0.1;
			double Evaluate(double X) const {
				return (1.0 / (std::sqrt(2.0 * PI * Variance))) * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
			}
		};

		struct GaussianMixtureModel {
			int K, N;
			std::vector<GaussianDistribution> GaussianDistributions;
			std::vector<double> MixtureCoefficients;
			std::vector<cv::Mat> PostProbility;
			std::vector<double>GCache;
			double LogLikehoodCache = 1;

			GaussianMixtureModel(const int K, const int N) :K(K),N(N),
				GaussianDistributions(std::vector<GaussianDistribution>(K)),
				MixtureCoefficients(std::vector<double>(K * size_t(N))),
				PostProbility(std::vector<cv::Mat>(K)),
				GCache(std::vector<double>(K * size_t(N))){
			}

			void Save(const char* Filename) {
				FILE* File = fopen(Filename, "wb");
				auto DistSize = GaussianDistributions.size();
				fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
				fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);
				auto MixSize = MixtureCoefficients.size();
				fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
				fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
				fclose(File);
			}

			void Load(const char* Filename) {
				FILE* File = fopen(Filename, "rb");
				size_t DistSize = 0, MixSize = 0;

				fread(&DistSize, sizeof(size_t), 1, File);
				GaussianDistributions.resize(DistSize);
				fread(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);

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

			GaussianDistribution& GetGaussianDistribution(int Index) {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			const GaussianDistribution& GetGaussianDistribution(int Index) const {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			void WritePostProbility(const char* FilePrefix = "PostProb") {
				printf("Write PostProbility....\n");
				char FilenameBuffer[256];
				for (int i = 0; i < K; i++) {
					std::memset(FilenameBuffer, 0, sizeof(FilenameBuffer));
					sprintf(FilenameBuffer, "%s_%d.txt", FilePrefix, i);
					FILE* F = fopen(FilenameBuffer, "w");
					for (int j = 0; j < N; j++) {
						fprintf(F, "%.3f ", PostProbility[i].at<double>(j));
					}
					fprintf(F, "\n");
					fclose(F);
				}
			}

			std::string ToString() {
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
			//////////////// ALGORITHM METHOD ////////////////
			void Initialize(const cv::Mat& InImage) {
				printf("Initialize....\n");

				//std::vector<double>Means;
				//KMeans(InImage, K, Means);
				//for (int i = 0; i < Count(); i++) {
				//	GetGaussianDistribution(i).Expectation = Means[i];
				//}

				////std::srand(std::time(NULL));
				for (int i = 0; i < Count(); i++) {
					int RandRow = std::rand() % InImage.rows;
					int RandCol = std::rand() % InImage.cols;
					GetGaussianDistribution(i).Expectation = InImage.at<double>(RandRow, RandCol);
				}

				std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

				for (int i = 0; i < PostProbility.size(); i++) {
					PostProbility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
				}
			}

			void EStep(const cv::Mat& InputImage) {
				printf(">>>>>>>>>>>>>>>> E-Step....\n");
				for (int i = 0; i < Count(); i++) {
					printf("Estimate %d/%d\n", i + 1, Count());
					for (int NIndex = 0; NIndex < N; NIndex++) {
						double Up = GetMixtureCoefficient(i, NIndex) * GaussianDistributions[i].Evaluate(InputImage.at<double>(NIndex));
						double Sum = 1e-9;
						for (int j = 0; j < Count(); j++) {
							Sum += GetMixtureCoefficient(j, NIndex) * GaussianDistributions[j].Evaluate(InputImage.at<double>(NIndex));
						}
						PostProbility[i].at<double>(NIndex) = Up / Sum;
					}
				}
			}

			void MStep(const cv::Mat& InImage, std::vector<double>& DExp, std::vector<double>& DVar,
				double& DLogLikehood) {
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
						GaussianDistribution NewGaussianDistrib;

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

						DExp[i] = std::abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
						DVar[i] = std::abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);

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
					if(LogLikehoodCache <= 0) {
						DLogLikehood = std::abs(LogLikehood - LogLikehoodCache);
					}
					LogLikehoodCache = LogLikehood;
					printf("%f\n", LogLikehood);
				}
			}
		
		};

		struct Context {
			int IterationCount = 0;
			std::vector<double>DExp;
			std::vector<double>DVar;
			double DLogLikehood;

			Context(int K) :
				DExp(std::vector<double>(K, DBL_MAX)),
				DVar(std::vector<double>(K, DBL_MAX)),
				DLogLikehood(DBL_MAX){
			}

			double MaxDExp() const {
				return MaxValue(DExp);
			}

			double MaxDVar() const {
				return MaxValue(DVar);
			}

			bool IsSatisfied() {
				return IterationCount >= 20
					|| (MaxDExp() <= 0.001
						&& MaxDVar() <= 0.001
						&& DLogLikehood <= 100);
			}

			std::string ToString() {
				std::stringstream Stream;
				Stream << "Iteration\t: " << IterationCount << "\n"
					<< "Max DExp\t: " << MaxDExp() << "\n"
					<< "Max DVar\t: " << MaxDVar() << "\n"
					<< "DLogLikehood\t: " << DLogLikehood << "\n";
				return Stream.str();
			}
		private:
			double MaxValue(const std::vector<double>& Vec) const {
				return *std::max_element(Vec.begin(), Vec.end());
			}
		};

		////////////////// Implementation ////////////////
		//	Step 1) Initialize the parameters: the means μj,
		//			covariance values j, and prior distributions πij .
		//	Step 2) E step.
		//			a) Evaluate the values zij in(17) using the current
		//			parameter values.
		//			b) Update the factor Gij by using (12).
		//	Step 3) M step : Reestimate the parameters.
		//			a) Update the means μj by using (21).
		//			b) Update covariance values j by using (23).
		//			c) Update prior distributions πij by using (27).
		//	Step 4) Evaluate the log - likelihood in(15) and check the
		//			convergence of either the log - likelihood function or
		//			the parameter values.If the convergence criterion is
		//			not satisfied, then go to step 2.

		//// begin
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		GaussianMixtureModel Model(K, InImage.rows * InImage.cols);
		Context Context(K);

		//// 初始化
		Model.Initialize(InImage);

		if (Arg.mInputModel) {
			printf("Load model %s\n", Arg.mInputModel);
			Model.Load(Arg.mInputModel);
		}
		else {
			printf("-------- ModifiedGMMSegmentation BEGIN --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
			// 迭代求解
			while (!Context.IsSatisfied()) {
				printf("======== Iter #%d ========\n", Context.IterationCount + 1);
				// E-Step
				Model.EStep(InImage);
				// M-Step
				Model.MStep(InImage, Context.DExp, Context.DVar, Context.DLogLikehood);
				// Update Context
				Context.IterationCount++;
				printf("Model: \n%s", Model.ToString().c_str());
				printf("Context: \n%s\n", Context.ToString().c_str());
			}
			printf("-------- ModifiedGMMSegmentation End --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
		}

		if (Arg.mOutputModel) {
			printf("Save model to %s\n", Arg.mOutputModel);
			Model.Save(Arg.mOutputModel);
		}

		////////////////// Segmenation ////////////////
		Model.EStep(InImage);
		double Step = 1.0 / (double(K) - 1);
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MaxProbility = 0.0;
			int MaxI = 0;
			auto& Probility = Model.PostProbility;
			for (int i = 0; i < Probility.size(); i++) {
				if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
					MaxI = i;
					MaxProbility = Probility[i].at<double>(Position);
				}
			}
			Pixel = Step * MaxI;
		}
		);

		//Model.WritePostProbility();
	}

	static void ModifiedGMMSegmentationColor(const cv::Mat& InImage, cv::Mat& OutImage, const SegArg& Arg) {
		int K = Arg.mComponentCount;
		assert(K > 1);
		//////////////// Type Def ////////////////
		struct GaussianDistribution {
			cv::Vec3d Expectation = { 0,0,0 };
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
				Cache1 = 1.0 / (std::sqrt(std::pow(2.0 * PI, 3) * cv::determinant(VarianceMat)));
				Cache2 = VarianceMat.inv();
			}
		};

		struct GaussianMixtureModel {
			int K, N;
			std::vector<GaussianDistribution> GaussianDistributions;
			std::vector<double> MixtureCoefficients;
			std::vector<cv::Mat> PostProbility;
			std::vector<double>GCache;
			double LogLikehoodCache = 1;

			GaussianMixtureModel(const int K, const int N) :K(K), N(N),
				GaussianDistributions(std::vector<GaussianDistribution>(K)),
				MixtureCoefficients(std::vector<double>(K* size_t(N))),
				PostProbility(std::vector<cv::Mat>(K)),
				GCache(std::vector<double>(K* size_t(N))) {
			}

			void Save(const char* Filename) {
				FILE* File = fopen(Filename, "wb");
				auto DistSize = GaussianDistributions.size();
				fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
				fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);
				auto MixSize = MixtureCoefficients.size();
				fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
				fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
				fclose(File);
			}

			void Load(const char* Filename) {
				FILE* File = fopen(Filename, "rb");
				size_t DistSize = 0, MixSize = 0;

				fread(&DistSize, sizeof(size_t), 1, File);
				GaussianDistributions.resize(DistSize);
				fread(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);

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

			GaussianDistribution& GetGaussianDistribution(int Index) {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			const GaussianDistribution& GetGaussianDistribution(int Index) const {
				assert(0 <= Index && Index < Count());
				return GaussianDistributions[Index];
			}

			void WritePostProbility(const char* FilePrefix = "PostProb") {
				printf("Write PostProbility....\n");
				char FilenameBuffer[256];
				for (int i = 0; i < K; i++) {
					std::memset(FilenameBuffer, 0, sizeof(FilenameBuffer));
					sprintf(FilenameBuffer, "%s_%d.txt", FilePrefix, i);
					FILE* F = fopen(FilenameBuffer, "w");
					for (int j = 0; j < N; j++) {
						fprintf(F, "%.3f ", PostProbility[i].at<double>(j));
					}
					fprintf(F, "\n");
					fclose(F);
				}
			}

			std::string ToString() {
				std::stringstream Stream;
				const char* Format = "#%d Exp: %s\tVar: %s\n";
				char Buffer[256];
				for (int i = 0; i < Count(); i++) {
					memset(Buffer, 0, sizeof(Buffer));
					sprintf_s(Buffer, Format, i, 
						ToString(GaussianDistributions[i].Expectation).c_str(), 
						ToString(GaussianDistributions[i].VarianceMat).c_str());
					Stream << Buffer;
				}
				return Stream.str();
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
			//////////////// ALGORITHM METHOD ////////////////
			void Initialize(const cv::Mat& InImage) {
				printf("Initialize....\n");

				std::vector<cv::Vec3d>Means;
				KMeansColor(InImage, K, Means);
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).Expectation = Means[i];
				}

				////std::srand(std::time(NULL));
				//for (int i = 0; i < Count(); i++) {
				//	int RandRow = std::rand() % InImage.rows;
				//	int RandCol = std::rand() % InImage.cols;
				//	GetGaussianDistribution(i).Expectation = InImage.at<cv::Vec3d>(RandRow, RandCol);
				//}

				std::fill(MixtureCoefficients.begin(), MixtureCoefficients.end(), 1.0 / K);

				for (int i = 0; i < PostProbility.size(); i++) {
					PostProbility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
				}
			}

			void EStep(const cv::Mat& InputImage) {
				this->UpdateCache();
				printf(">>>>>>>>>>>>>>>> E-Step....\n");
				for (int i = 0; i < Count(); i++) {
					printf("Estimate %d/%d\n", i + 1, Count());
#pragma omp parallel for
					for (int NIndex = 0; NIndex < N; NIndex++) {
						double Up = GetMixtureCoefficient(i, NIndex) * GaussianDistributions[i].Evaluate(InputImage.at<cv::Vec3d>(NIndex));
						double Sum = 1e-9;
						for (int j = 0; j < Count(); j++) {
							Sum += GetMixtureCoefficient(j, NIndex) * GaussianDistributions[j].Evaluate(InputImage.at<cv::Vec3d>(NIndex));
						}
						PostProbility[i].at<double>(NIndex) = Up / Sum;
					}
				}
			}

			void MStep(const cv::Mat& InImage, std::vector<double>& DExp, std::vector<double>& DVar,
				double& DLogLikehood) {
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
						GaussianDistribution NewGaussianDistrib;

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

						DExp[i] = Vec3dAbs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
						DVar[i] = MatAbs(OldGaussianDistrib.VarianceMat - NewGaussianDistrib.VarianceMat);

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
						DLogLikehood = std::abs(LogLikehood - LogLikehoodCache);
					}
					LogLikehoodCache = LogLikehood;
					printf("%f\n", LogLikehood);
				}
			}

		private:
			//////////////// UTILITY ////////////////
			void UpdateCache() {
				// update cache
				for (int i = 0; i < Count(); i++) {
					GetGaussianDistribution(i).UpdateCache();
				}
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
				return Sum / (Input.rows * Input.cols);
			};
		};

		struct Context {
			int IterationCount = 0;
			std::vector<double>DExp;
			std::vector<double>DVar;
			double DLogLikehood;

			Context(int K) :
				DExp(std::vector<double>(K, DBL_MAX)),
				DVar(std::vector<double>(K, DBL_MAX)),
				DLogLikehood(DBL_MAX) {
			}

			double MaxDExp() const {
				return MaxValue(DExp);
			}

			double MaxDVar() const {
				return MaxValue(DVar);
			}

			bool IsSatisfied() {
				return IterationCount >= 20
					|| (MaxDExp() <= 0.001
						&& MaxDVar() <= 0.001
						&& DLogLikehood <= 100);
			}

			std::string ToString() {
				std::stringstream Stream;
				Stream << "Iteration\t: " << IterationCount << "\n"
					<< "Max DExp\t: " << MaxDExp() << "\n"
					<< "Max DVar\t: " << MaxDVar() << "\n"
					<< "DLogLikehood\t: " << DLogLikehood << "\n";
				return Stream.str();
			}
		private:
			double MaxValue(const std::vector<double>& Vec) const {
				return *std::max_element(Vec.begin(), Vec.end());
			}
		};

		////////////////// Implementation ////////////////
		//	Step 1) Initialize the parameters: the means μj,
		//			covariance values j, and prior distributions πij .
		//	Step 2) E step.
		//			a) Evaluate the values zij in(17) using the current
		//			parameter values.
		//			b) Update the factor Gij by using (12).
		//	Step 3) M step : Reestimate the parameters.
		//			a) Update the means μj by using (21).
		//			b) Update covariance values j by using (23).
		//			c) Update prior distributions πij by using (27).
		//	Step 4) Evaluate the log - likelihood in(15) and check the
		//			convergence of either the log - likelihood function or
		//			the parameter values.If the convergence criterion is
		//			not satisfied, then go to step 2.

		//// begin
		OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		GaussianMixtureModel Model(K, InImage.rows * InImage.cols);
		Context Context(K);

		//// 初始化
		Model.Initialize(InImage);

		if (Arg.mInputModel) {
			printf("Load model %s\n", Arg.mInputModel);
			Model.Load(Arg.mInputModel);
		}
		else {
			printf("-------- ModifiedGMMSegmentation BEGIN --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
			// 迭代求解
			while (!Context.IsSatisfied()) {
				printf("======== Iter #%d ========\n", Context.IterationCount + 1);
				// E-Step
				Model.EStep(InImage);
				// M-Step
				Model.MStep(InImage, Context.DExp, Context.DVar, Context.DLogLikehood);
				// Update Context
				Context.IterationCount++;
				printf("Model: \n%s", Model.ToString().c_str());
				printf("Context: \n%s\n", Context.ToString().c_str());
			}
			printf("-------- ModifiedGMMSegmentation End --------\n");
			printf("Model: \n%s", Model.ToString().c_str());
			printf("Context: \n%s\n", Context.ToString().c_str());
		}

		if (Arg.mOutputModel) {
			printf("Save model to %s\n", Arg.mOutputModel);
			Model.Save(Arg.mOutputModel);
		}

		////////////////// Segmenation ////////////////
		Model.EStep(InImage);
		double Step = 1.0 / (double(K) - 1);
		OutImage.forEach<double>(
			[&](double& Pixel, const int* Position) {
			double MaxProbility = 0.0;
			int MaxI = 0;
			auto& Probility = Model.PostProbility;
			for (int i = 0; i < Probility.size(); i++) {
				if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
					MaxI = i;
					MaxProbility = Probility[i].at<double>(Position);
				}
			}
			Pixel = Step * MaxI;
		}
		);

		//Model.WritePostProbility();
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

		switch (Arg.mSegType)
		{
		case SegArg::ESegType::GMMColor:
			GMMSegmentationColor(InputImage, SegmentedImg, Arg);
			break;
		case SegArg::ESegType::GMMGray:
			GMMSegmentationGray(InputImage, SegmentedImg, Arg);
			break;
		case SegArg::ESegType::MGMMColor:
			ModifiedGMMSegmentationColor(InputImage, SegmentedImg, Arg);
			break;
		case SegArg::ESegType::MGMMGray:
			ModifiedGMMSegmentationGray(InputImage, SegmentedImg, Arg);
			break;
		case SegArg::ESegType::KMeansColor:
			KMeansSegmentationColor(InputImage, SegmentedImg, Arg);
			break;
		case SegArg::ESegType::KMeansGray:
			KMeansSegmentationGray(InputImage, SegmentedImg, Arg);
			break;
		default:
			break;
		}

		if (!OK) {
			printf("read image fail\n");
		}
		else {
			CV::DisplayImage(InputImage, "Origin");
			CV::DisplayImage(SegmentedImg, "Segmented");
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

		TestSegmentation(
			TestData.back().first,
			SegArg()
				.RandomSeed(true)
				.InputModel(nullptr)
				.DCoeThreshold(0.001)
				.DExpThreshold(0.001)
				.DVarThreshold(0.001)
				.OutputModel(nullptr)
				.MaxIterationCount(20)
				.KMeansInitialized(true)
				.DLogLikehoodThreshold(100)
				.SegType(SegArg::ESegType::GMMGray)
				.SegType(SegArg::ESegType::GMMColor)
				.SegType(SegArg::ESegType::MGMMGray)
				.SegType(SegArg::ESegType::MGMMColor)
				.SegType(SegArg::ESegType::KMeansGray)
				.SegType(SegArg::ESegType::KMeansColor)
				.ComponentCount(TestData.back().second)
		);

		CV::Wait();
	}
}