#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
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

	static void Wait(int Delay=0) {
		cv::waitKey(Delay);
	}
}

namespace GMM {
	static void GMMSegmentation(const cv::Mat& InImage, const int K, cv::Mat& OutImage,
		const char* OutputModel = nullptr, const char* InputModel = nullptr) {
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
			return Context.IterationCount >= 10
				|| (Context.MaxDCoeff() <= 0.001
					&& Context.MaxDExp() <= 0.001
					&& Context.MaxDVar() <= 0.001);
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

		if (InputModel) {
			printf("Load model %s\n", InputModel);
			Model.Load(InputModel);
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

		if (OutputModel) {
			printf("Save model to %s\n", OutputModel);
			Model.Save(OutputModel);
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

	static void ModifiedGMMSegmentation(const cv::Mat& InImage, const int K, cv::Mat& OutImage,
		const char* OutputModel = nullptr, const char* InputModel = nullptr) {
		assert(K > 1);
		//////////////// Type Def ////////////////
		struct GaussianDistribution {
			double Expectation = 0;
			double Variance = 0.1;
			double Evaluate(double X) const {
				return (1.0 / (std::sqrt(2.0 * PI * Variance))) * std::pow(E, -0.5 * std::pow(X - Expectation, 2) / Variance);
			}
		};

		//struct GaussianMixtureModel {
		//	std::vector<GaussianDistribution> GaussianDistributions;
		//	std::vector<double> MixtureCoefficients;

		//	GaussianMixtureModel(const int K) :
		//		GaussianDistributions(std::vector<GaussianDistribution>(K)),
		//		MixtureCoefficients(std::vector<double>(K)) {
		//	}

		//	void Save(const char* Filename) {
		//		FILE* File = fopen(Filename, "wb");
		//		auto DistSize = GaussianDistributions.size();
		//		fwrite(&DistSize, sizeof(GaussianDistributions.size()), 1, File);
		//		fwrite(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);
		//		auto MixSize = MixtureCoefficients.size();
		//		fwrite(&MixSize, sizeof(MixtureCoefficients.size()), 1, File);
		//		fwrite(&MixtureCoefficients[0], sizeof(double), MixSize, File);
		//		fclose(File);
		//	}

		//	void Load(const char* Filename) {
		//		FILE* File = fopen(Filename, "rb");
		//		size_t DistSize = 0, MixSize = 0;

		//		fread(&DistSize, sizeof(size_t), 1, File);
		//		GaussianDistributions.resize(DistSize);
		//		fread(&GaussianDistributions[0], sizeof(GaussianDistribution), DistSize, File);

		//		fread(&MixSize, sizeof(size_t), 1, File);
		//		MixtureCoefficients.resize(MixSize);
		//		fread(&MixtureCoefficients[0], sizeof(double), MixSize, File);

		//		fclose(File);
		//	}

		//	int Count() const {
		//		return GaussianDistributions.size();
		//	}

		//	double GetMixtureCoefficient(int Index) const {
		//		assert(0 <= Index && Index < Count());
		//		return MixtureCoefficients[Index];
		//	}

		//	double& GetMixtureCoefficient(int Index) {
		//		assert(0 <= Index && Index < Count());
		//		return MixtureCoefficients[Index];
		//	}

		//	GaussianDistribution& GetGaussianDistribution(int Index) {
		//		assert(0 <= Index && Index < Count());
		//		return GaussianDistributions[Index];
		//	}

		//	const GaussianDistribution& GetGaussianDistribution(int Index) const {
		//		assert(0 <= Index && Index < Count());
		//		return GaussianDistributions[Index];
		//	}

		//	std::string ToString() {
		//		std::stringstream Stream;
		//		const char* Format = "#%d Coe: %.6f\tExp: %.6f\tVar: %.6f\n";
		//		char Buffer[256];
		//		for (int i = 0; i < Count(); i++) {
		//			memset(Buffer, 0, sizeof(Buffer));
		//			sprintf_s(Buffer, Format, i, MixtureCoefficients[i], GaussianDistributions[i].Expectation, GaussianDistributions[i].Variance);
		//			Stream << Buffer;
		//		}
		//		Stream << "\n";
		//		return Stream.str();
		//	}
		//};

		//struct Context {
		//	int IterationCount = 0;
		//	std::vector<double>DCoeff;
		//	std::vector<double>DExp;
		//	std::vector<double>DVar;

		//	Context(int K) :
		//		DCoeff(std::vector<double>(K, DBL_MAX)),
		//		DExp(std::vector<double>(K, DBL_MAX)),
		//		DVar(std::vector<double>(K, DBL_MAX)) {
		//	}

		//	double MaxDCoeff() const {
		//		return MaxValue(DCoeff);
		//	}

		//	double MaxDExp() const {
		//		return MaxValue(DExp);
		//	}

		//	double MaxDVar() const {
		//		return MaxValue(DVar);
		//	}

		//	std::string ToString() {
		//		std::stringstream Stream;
		//		Stream << "Iteration\t: " << IterationCount << "\n"
		//			<< "Max DCoeff\t: " << MaxDCoeff() << "\n"
		//			<< "Max DExp\t: " << MaxDExp() << "\n"
		//			<< "Max DVar\t: " << MaxDVar() << "\n";
		//		return Stream.str();
		//	}
		//private:
		//	double MaxValue(const std::vector<double>& Vec) const {
		//		return *std::max_element(Vec.begin(), Vec.end());
		//	}
		//};
		////////////////// Method Def ////////////////
		//// 初始化
		//static auto Initialize = [](GaussianMixtureModel& Model, std::vector<cv::Mat>& Probility, const cv::Mat& InImage) {
		//	for (int i = 0; i < Model.Count(); i++) {
		//		Model.MixtureCoefficients[i] = 1.0 / Model.Count();
		//		int RandRow = std::rand() % InImage.rows;
		//		int RandCol = std::rand() % InImage.cols;
		//		Model.GetGaussianDistribution(i).Expectation = InImage.at<double>(RandRow, RandCol);
		//	}

		//	for (int i = 0; i < Probility.size(); i++) {
		//		Probility[i] = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		//	}
		//};

		//// 是否满足停止条件
		//static auto CheckCondition = [](const Context& Context) -> bool {
		//	return Context.IterationCount >= 10
		//		|| (Context.MaxDCoeff() <= 0.001
		//			&& Context.MaxDExp() <= 0.001
		//			&& Context.MaxDVar() <= 0.001);
		//};

		//// 计算概率
		//static auto ComputeProbability = [](const cv::Mat& InputImage, const GaussianMixtureModel& Model, std::vector<cv::Mat>& OutProbility) {
		//	for (int i = 0; i < Model.Count(); i++) {
		//		for (int row = 0; row < OutProbility[i].rows; row++) {
		//			for (int col = 0; col < OutProbility[i].cols; col++) {
		//				double Up = Model.GetMixtureCoefficient(i) * Model.GetGaussianDistribution(i).Evaluate(InputImage.at<double>(row, col));
		//				double Sum = 0;
		//				for (int j = 0; j < Model.Count(); j++) {
		//					Sum += Model.GetMixtureCoefficient(j) * Model.GetGaussianDistribution(j).Evaluate(InputImage.at<double>(row, col));
		//				}
		//				OutProbility[i].at<double>(row, col) = Up / Sum;
		//			}
		//		}
		//	}
		//};

		//// 更新参数
		//static auto UpdateParameters = [](GaussianMixtureModel& Model, const cv::Mat& InImage, const std::vector<cv::Mat>& InProbility,
		//	std::vector<double>& DCoeff, std::vector<double>& DExp, std::vector<double>& DVar) {
		//	for (int i = 0; i < Model.Count(); i++) {
		//		double SumProbility = 0.0;
		//		double SumExpectation = 0.0;
		//		for (int row = 0; row < InProbility[i].rows; row++) {
		//			for (int col = 0; col < InProbility[i].cols; col++) {
		//				SumProbility += InProbility[i].at<double>(row, col);
		//				SumExpectation += InProbility[i].at<double>(row, col) * InImage.at<double>(row, col);
		//			}
		//		}
		//		auto N = InProbility[i].rows * InProbility[i].cols;
		//		auto& OldCoeff = Model.GetMixtureCoefficient(i);
		//		auto& OldGaussianDistrib = Model.GetGaussianDistribution(i);
		//		auto NewCoeff = SumProbility / N;
		//		GaussianDistribution NewGaussianDistrib;
		//		NewGaussianDistrib.Expectation = SumExpectation / SumProbility;

		//		double SumVariance = 0.0;
		//		for (int row = 0; row < InProbility[i].rows; row++) {
		//			for (int col = 0; col < InProbility[i].cols; col++) {
		//				SumVariance += InProbility[i].at<double>(row, col) * std::pow(InImage.at<double>(row, col) - NewGaussianDistrib.Expectation, 2);
		//			}
		//		}
		//		NewGaussianDistrib.Variance = SumVariance / SumProbility;

		//		DCoeff[i] = std::abs(OldCoeff - NewCoeff);
		//		DExp[i] = std::abs(OldGaussianDistrib.Expectation - NewGaussianDistrib.Expectation);
		//		DVar[i] = std::abs(OldGaussianDistrib.Variance - NewGaussianDistrib.Variance);

		//		OldCoeff = NewCoeff;
		//		OldGaussianDistrib = NewGaussianDistrib;
		//	}
		//};

		////////////////// Implementation ////////////////
		//// 初始化
		//OutImage = cv::Mat(InImage.rows, InImage.cols, CV_64FC1);
		//GaussianMixtureModel Model(K);
		//std::vector<cv::Mat> Probility(K);
		//Context Context(K);
		//Initialize(Model, Probility, InImage);

		//if (InputModel) {
		//	printf("Load model %s\n", InputModel);
		//	Model.Load(InputModel);
		//}
		//else {
		//	printf("-------- BEGIN --------\n");
		//	printf("Model: \n%s", Model.ToString().c_str());
		//	printf("Context: \n%s", Context.ToString().c_str());
		//	// 迭代求解
		//	while (!CheckCondition(Context)) {
		//		// E-Step
		//		ComputeProbability(InImage, Model, Probility);
		//		// M-Step
		//		UpdateParameters(Model, InImage, Probility,
		//			Context.DCoeff, Context.DExp, Context.DVar);
		//		// Update Context
		//		Context.IterationCount++;
		//	}
		//	printf("-------- End --------\n");
		//	printf("Model: \n%s", Model.ToString().c_str());
		//	printf("Context: \n%s", Context.ToString().c_str());
		//}

		//if (OutputModel) {
		//	printf("Save model to %s\n", OutputModel);
		//	Model.Save(OutputModel);
		//}

		////////////////// Segmenation ////////////////
		//ComputeProbability(InImage, Model, Probility);
		//double Step = 1.0 / double(K - 1);
		//OutImage.forEach<double>(
		//	[&](double& Pixel, const int* Position) {
		//	double MaxProbility = 0.0;
		//	int MaxI = 0;
		//	for (int i = 0; i < Probility.size(); i++) {
		//		if (i == 0 || MaxProbility < Probility[i].at<double>(Position)) {
		//			MaxI = i;
		//			MaxProbility = Probility[i].at<double>(Position);
		//		}
		//	}
		//	Pixel = Step * MaxI;
		//}
		//);
	}

	static void TestGMMSegmentation(const char* InputImageName, const char* OutputModel = nullptr, const char* InputModel = nullptr) {
		cv::Mat InputImage;
		if (CV::ReadImage(InputImageName, InputImage, cv::IMREAD_GRAYSCALE)) {
			InputImage.convertTo(InputImage, CV_64FC1, 1.0 / 255.0);

			cv::Mat SegmentedImg;
			GMMSegmentation(InputImage, 4, SegmentedImg, OutputModel, InputModel);

			CV::DisplayImage(InputImage, "Origin");
			CV::DisplayImage(SegmentedImg, "Segmented");
		}
		else {
			printf("read image fail\n");
		}
	}

	static void TestModifiedGMMSegmentation(const char* InputImageName, const char* OutputModel = nullptr, const char* InputModel = nullptr) {
		cv::Mat InputImage;
		if (CV::ReadImage(InputImageName, InputImage, cv::IMREAD_GRAYSCALE)) {
			InputImage.convertTo(InputImage, CV_64FC1, 1.0 / 255.0);

			cv::Mat SegmentedImg;
			ModifiedGMMSegmentation(InputImage, 4, SegmentedImg, OutputModel, InputModel);

			CV::DisplayImage(InputImage, "Origin");
			CV::DisplayImage(SegmentedImg, "Segmented");
		}
		else {
			printf("read image fail\n");
		}
	}

	static void Main() {
		const char* InputImageName0 = "C:\\Users\\35974\\Pictures\\Saved Pictures\\blog\\002\\0.png";
		const char* InputImageName1 = "C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\leaf.jpg";
		const char* InputImageName2 = "C:\\Users\\35974\\Pictures\\Saved Pictures\\Study\\Lenna.jpg";
		const char* InputImageName3 = "D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f02970.png";
		const char* InputImageName4 = "D:\\Study\\毕业设计\\Dataset\\CamVid11\\CamVid\\images\\test\\Seq05VD_f00300.png";
		//TestGMMSegmentation(InputImageName2);
		TestModifiedGMMSegmentation(InputImageName2);
		CV::Wait();
	}
}