#include "cob_texture_categorization/train_ml.h"

#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <ctype.h>
#include <cv.h>


#include "ml.h"
#include "highgui.h"

train_ml::train_ml()
{
}

void train_ml::load_texture_database_features(std::string path, cv::Mat& feature_matrix, cv::Mat& label_matrix, create_train_data::DataHierarchyType& data_sample_hierarchy)
{
	// load feature vectors and corresponding labels computed on database
	std::string database_file = path + "database_data.yml";
	cv::FileStorage fs(database_file, cv::FileStorage::READ);
	fs["database_data"] >> feature_matrix;

	std::string database_label_file = path + "database_label.yml";
	cv::FileStorage fsl(database_label_file, cv::FileStorage::READ);
	fsl["database_label"] >> label_matrix;

	// load class-object-sample hierarchy
	std::string database_hierarchy_file = path + "data_hierarchy.txt";
	create_train_data data_object;
	data_object.load_data_hierarchy(database_hierarchy_file, data_sample_hierarchy);
}

void train_ml::cross_validation(int folds, const cv::Mat& feature_matrix, const cv::Mat& label_matrix, const create_train_data::DataHierarchyType& data_sample_hierarchy)
{
	std::vector<int> true_predictions, false_predictions;
	create_train_data data_object;
	std::vector<std::string> texture_classes = data_object.get_texture_classes();

	for (int fold=0; fold<folds; ++fold)
	{
		std::cout << "=== fold " << fold << " ===" << std::endl;

		// === distribute data into training and test set ===
		// select one object per class for testing
		srand(0);	// random seed --> keep reproducible
		std::vector<int> train_indices, test_indices;
		for (unsigned int class_index=0; class_index<data_sample_hierarchy.size(); ++class_index)
		{
			int object_number = data_sample_hierarchy[class_index].size();
			int test_object = (int)(object_number * (double)rand()/((double)RAND_MAX+1.0));
//			std::cout << "object_number=" << object_number << "   test_object=" << test_object << std::endl;
			for (int object_index=0; object_index<object_number; ++object_index)
			{
				if (object_index == test_object)
					for (unsigned int s=0; s<data_sample_hierarchy[class_index][object_index].size(); ++s)
						test_indices.push_back(data_sample_hierarchy[class_index][object_index][s]);
				else
					for (unsigned int s=0; s<data_sample_hierarchy[class_index][object_index].size(); ++s)
						train_indices.push_back(data_sample_hierarchy[class_index][object_index][s]);
			}
		}
		assert(test_indices.size() + train_indices.size() == feature_matrix.rows);

		// create training and test data matrices
		cv::Mat training_data(train_indices.size(), feature_matrix.cols, feature_matrix.type());
		cv::Mat training_labels(train_indices.size(), 1, label_matrix.type());
		cv::Mat test_data(test_indices.size(), feature_matrix.cols, feature_matrix.type());
		cv::Mat test_labels(test_indices.size(), 1, label_matrix.type());
		for (unsigned int r=0; r<train_indices.size(); ++r)
		{
			for (int c=0; c<feature_matrix.cols; ++c)
				training_data.at<float>(r,c) = feature_matrix.at<float>(train_indices[r],c);
			training_labels.at<float>(r) = label_matrix.at<float>(train_indices[r]);
		}
		for (unsigned int r=0; r<test_indices.size(); ++r)
		{
			for (int c=0; c<feature_matrix.cols; ++c)
				test_data.at<float>(r,c) = feature_matrix.at<float>(test_indices[r],c);
			test_labels.at<float>(r) = label_matrix.at<float>(test_indices[r]);
		}

		// === train ml classifier ===

//	std::cout<<"Value:"<<test_data_label.at<float>(3,0)<<std::endl;
//	K-Nearest-Neighbor

//	cv::Mat test;
//	CvKNearest knn(training_data, training_label, cv::Mat(), false, 32);
//	knn.train(training_data,training_label,test,false,32,false );
//
//	cv::Mat result;
//	cv::Mat neighbor_results;
//	cv::Mat dist;
//	knn.find_nearest(train_data, 1, &result,0,&neighbor_results, &dist);

// 	End K-Nearest-Neighbor

//	Randomtree

//	float v1 = 0.0;
//	const float *v2 = 0;
//	float v3 = 1.0;
//	CvRTParams treetest(78, 6, v1, false,28, v2, false,23,28,v3, 0 );
//	CvRTrees tree;
//	tree.train(training_data,1, training_label, cv::Mat(), cv::Mat(),cv::Mat(),cv::Mat(), treetest);
//	cv::Mat result(train_data.rows,1,CV_32FC1);
//
//	for(int i=0;i<train_data.rows;i++)
//	{
//		cv::Mat inputvec(1,train_data.cols, CV_32FC1);
//		for(int j=0;j<train_data.cols;j++)
//		{
//			inputvec.at<float>(0,j)=train_data.at<float>(i,j);
//		}
//		result.at<float>(i,0)=tree.predict(inputvec);
//	}

//	End Randomtree

//	Decision Tree

//    CvDTreeParams params = CvDTreeParams(25, // max depth
//                                             5, // min sample count
//                                             0, // regression accuracy: N/A here
//                                             false, // compute surrogate split, no missing data
//                                             15, // max number of categories (use sub-optimal algorithm for larger numbers)
//                                             15, // the number of cross-validation folds
//                                             false, // use 1SE rule => smaller tree
//                                             false, // throw away the pruned tree branches
//                                             NULL // the array of priors
//                                            );
//    CvDTree dtree;
//    dtree.train(training_data, CV_ROW_SAMPLE, training_label, cv::Mat(), cv::Mat(),cv::Mat(),cv::Mat(), params);
//
//    CvDTreeNode *node;
//
//    cv::Mat result(train_data.rows,1,CV_32FC1);
//    	for(int i=0;i<train_data.rows;i++)
//    	{
//    		cv::Mat inputvec(1,train_data.cols, CV_32FC1);
//    		for(int j=0;j<train_data.cols;j++)
//    		{
//    			inputvec.at<float>(0,j)=train_data.at<float>(i,j);
//    		}
//    		node = dtree.predict(inputvec);
//    		result.at<float>(i,0)= (*node).class_idx;
//    	}

//	End Dicision Tree

//	BayesClassifier

//	CvNormalBayesClassifier bayesmod;
//	bayesmod.train(training_data, training_label, cv::Mat(), cv::Mat());
//	 cv::Mat result(train_data.rows,1,CV_32FC1);
//	bayesmod.predict(train_data, &result);

//	End Bayes Classifier

//	Neural Network

		cv::Mat input;
		training_data.convertTo(input, CV_32F);
		cv::Mat output=cv::Mat::zeros(training_data.rows, 57, CV_32FC1);

		cv::Mat labels;
		training_labels.convertTo(labels, CV_32F);

		for(int i=0; i<training_data.rows; ++i)
			output.at<float>(i,(int)labels.at<float>(i,0)) = 1.f;		//change.at<float>(i,0);

		cv::Mat layers = cv::Mat(3,1,CV_32SC1);

		layers.row(0) = cv::Scalar(16);
		layers.row(1) = cv::Scalar(400);
		layers.row(2) = cv::Scalar(57);

		CvANN_MLP mlp;
		CvANN_MLP_TrainParams params;
		CvTermCriteria criteria;

		criteria.max_iter = 1000;
		criteria.epsilon  = 0.00001f;
		criteria.type     = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

		params.train_method    = CvANN_MLP_TrainParams::BACKPROP;
		params.bp_dw_scale     = 0.1f;
		params.bp_moment_scale = 0.1f;
		params.term_crit       = criteria;

		mlp.create(layers,CvANN_MLP::SIGMOID_SYM,0.3,1.2);
		int iterations = mlp.train(input, output, cv::Mat(), cv::Mat(), params);
		std::cout << "Neural network training completed after " << iterations << " iterations." << std::endl;

		// === apply ml classifier to predict test set ===
		int t = 0, f = 0;
		int t2 = 0, f2 = 0;
		std::vector<int> labelres;

		for(int i = 0; i < test_data.rows ; i++)
		{
			cv::Mat response(1, 57, CV_32FC1);
			cv::Mat sample = test_data.row(i);

			mlp.predict(sample, response);

			float max = -1000000000000.0f;
			float max2 = -1000000000000.0f;
			int cls = -1;
			int cls2 = -1;
			for (int j = 0; j < 57; j++)
			{
				float value = response.at<float>(0, j);
				if (value > max)
				{
					max2 = max;
					cls2 = cls;
					max = value;
					cls = j;
				}
			}

			if (cls == test_labels.at<float>(i, 0))
				t++;
			else
				f++;

			if (cls2 == test_labels.at<float>(i, 0))
				t2++;
			else
				f2++;

			std::cout << "value:" << test_labels.at<float>(i, 0) << " (" << texture_classes[test_labels.at<float>(i, 0)] <<
					")   predicted:" << cls << " (" << texture_classes[cls] << ")" << std::endl;
		}

		true_predictions.push_back(t);
		false_predictions.push_back(f);
		std::cout << "true:" << t << " False:" << f << std::endl;
		std::cout << "true2:" << t2 << " False2:" << f2 << std::endl;
		double sum = t + f;
		double percentage = t*(100.0 / sum);
		std::cout << "Correct classified: " << percentage << "%" << std::endl;
	}
//	End Neural Network

	std::cout << "=== Total result over " << folds << "-fold cross validation ===" << std::endl;
	int t=0, f=0;
	for (unsigned int i=0; i<true_predictions.size(); ++i)
	{
		t += true_predictions[i];
		f += false_predictions[i];
	}
	std::cout << "Correct classified: " << t*100.0/(double)(t+f) << "%" << std::endl;
}


void train_ml::run_ml(double val, std::string *path_)
{

	std::string train_data = *path_ + "train_data.yml";
//	cv::FileStorage fs("/home/rmb-dh/Test_dataset/training_data.yml", cv::FileStorage::READ);
	cv::FileStorage fs(train_data, cv::FileStorage::READ);
	cv::Mat training_data;
	fs["train_data"] >> training_data;
//	std::cout<<training_data<<"Matrix"<<std::endl;

	std::string train_label = *path_ + "train_data_label.yml";
//	cv::FileStorage fsl("/home/rmb-dh/Test_dataset/train_data_respons.yml", cv::FileStorage::READ);
	cv::FileStorage fsl(train_label, cv::FileStorage::READ);
	cv::Mat training_labels;
	fsl["train_label"] >> training_labels;
//	std::cout<<training_label<<"Matrix"<<std::endl;

	cv::Mat test_data;
	std::string test_data_ = *path_ + "test_data.yml";
//	cv::FileStorage fsd("/home/rmb-dh/Test_dataset/test_data.yml", cv::FileStorage::READ);
	cv::FileStorage fsd(test_data_, cv::FileStorage::READ);
	fsd["test_data"] >> test_data;

//	std::vector<std::string> test_data_label;
	cv::Mat test_labels;
	std::string test_label = *path_ + "test_data_label.yml";
//	cv::FileStorage fstl("/home/rmb-dh/Test_dataset/test_data_label.yml", cv::FileStorage::READ);
	cv::FileStorage fstl(test_label, cv::FileStorage::READ);
	fstl["test_label"] >> test_labels;


	//	std::cout<<"Value:"<<test_data_label.at<float>(3,0)<<std::endl;
//	K-Nearest-Neighbor

//	cv::Mat test;
//	CvKNearest knn(training_data, training_label, cv::Mat(), false, 32);
//	knn.train(training_data,training_label,test,false,32,false );
//
//	cv::Mat result;
//	cv::Mat neighbor_results;
//	cv::Mat dist;
//	knn.find_nearest(train_data, 1, &result,0,&neighbor_results, &dist);

// 	End K-Nearest-Neighbor

//	Randomtree

//	float v1 = 0.0;
//	const float *v2 = 0;
//	float v3 = 1.0;
//	CvRTParams treetest(78, 6, v1, false,28, v2, false,23,28,v3, 0 );
//	CvRTrees tree;
//	tree.train(training_data,1, training_label, cv::Mat(), cv::Mat(),cv::Mat(),cv::Mat(), treetest);
//	cv::Mat result(train_data.rows,1,CV_32FC1);
//
//	for(int i=0;i<train_data.rows;i++)
//	{
//		cv::Mat inputvec(1,train_data.cols, CV_32FC1);
//		for(int j=0;j<train_data.cols;j++)
//		{
//			inputvec.at<float>(0,j)=train_data.at<float>(i,j);
//		}
//		result.at<float>(i,0)=tree.predict(inputvec);
//	}

//	End Randomtree

//	Decision Tree

//    CvDTreeParams params = CvDTreeParams(25, // max depth
//                                             5, // min sample count
//                                             0, // regression accuracy: N/A here
//                                             false, // compute surrogate split, no missing data
//                                             15, // max number of categories (use sub-optimal algorithm for larger numbers)
//                                             15, // the number of cross-validation folds
//                                             false, // use 1SE rule => smaller tree
//                                             false, // throw away the pruned tree branches
//                                             NULL // the array of priors
//                                            );
//    CvDTree dtree;
//    dtree.train(training_data, CV_ROW_SAMPLE, training_label, cv::Mat(), cv::Mat(),cv::Mat(),cv::Mat(), params);
//
//    CvDTreeNode *node;
//
//    cv::Mat result(train_data.rows,1,CV_32FC1);
//    	for(int i=0;i<train_data.rows;i++)
//    	{
//    		cv::Mat inputvec(1,train_data.cols, CV_32FC1);
//    		for(int j=0;j<train_data.cols;j++)
//    		{
//    			inputvec.at<float>(0,j)=train_data.at<float>(i,j);
//    		}
//    		node = dtree.predict(inputvec);
//    		result.at<float>(i,0)= (*node).class_idx;
//    	}

//	End Dicision Tree

//	BayesClassifier

//	CvNormalBayesClassifier bayesmod;
//	bayesmod.train(training_data, training_label, cv::Mat(), cv::Mat());
//	 cv::Mat result(train_data.rows,1,CV_32FC1);
//	bayesmod.predict(train_data, &result);

//	End Bayes Classifier

//	Neural Network

//		cv::Mat input;
//		training_data.convertTo(input, CV_32F);
//		cv::Mat output=cv::Mat::zeros(training_data.rows,57, CV_32FC1);

		cv::Mat input;
		training_data.convertTo(input, CV_32F);
		cv::Mat output=cv::Mat::zeros(training_data.rows,57, CV_32FC1);

		cv::Mat change;
		training_labels.convertTo(change, CV_32F);


		for(float i=0;i<training_data.rows;i++)
		{
			output.at<float>(i,(int)change.at<float>(i,0))=1;//change.at<float>(i,0);
		}


		 cv::Mat layers = cv::Mat(3,1,CV_32SC1);
//		    int sz = train_data.cols ;

		    layers.row(0) = cv::Scalar(16);
		    layers.row(1) = cv::Scalar(400);
		    layers.row(2) = cv::Scalar(57);

		    CvANN_MLP mlp;
		    CvANN_MLP_TrainParams params;
		    CvTermCriteria criteria;

		    criteria.max_iter = 1000;
		    criteria.epsilon  = 0.00001f;
		    criteria.type     = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

		    params.train_method    = CvANN_MLP_TrainParams::BACKPROP;
		    params.bp_dw_scale     = 0.1f;
		    params.bp_moment_scale = 0.1f;
		    params.term_crit       = criteria;

		    mlp.create(layers,CvANN_MLP::SIGMOID_SYM,0.3,1.2);
		    int i = mlp.train(input,output,cv::Mat(),cv::Mat(),params);

		    int t = 0, f = 0;
		    int t2 = 0, f2 = 0;

		    std::cout<<"training completed"<<std::endl;
		    std::vector<int> labelres;
		    for(int i = 0; i < test_data.rows ; i++)
		    {
		        cv::Mat response(1,57,CV_32FC1);
		        cv::Mat sample = test_data.row(i);

		        mlp.predict(sample,response);

		        float max = -1000000000000.0f;
		        float max2 = -1000000000000.0f;
		        int cls = -1;
		        int cls2 = -1;

		        for(int j = 0 ; j < 57 ; j++)
		        {
		            float value = response.at<float>(0,j);

		            if(value > max)
		            {
		            	max2=max;
		            	cls2=cls;
		                max = value;
		                cls = j;
		            }
		        }

		        if(cls == test_labels.at<float>(i,0))
		            t++;
		        else
		            f++;

		        if(cls2 == test_labels.at<float>(i,0))
		       		            t2++;
		       		        else
		       		            f2++;


//		        std::cout<<"test"<<std::endl;
		        std::cout<<"Value:"<<test_labels.at<float>(i,0)<<" Predicted:"<<cls<<std::endl;
		    }



		    std::cout<<"true:"<<t<<" False:"<<f<<std::endl;
		    std::cout<<"true2:"<<t2<<" False2:"<<f2<<std::endl;
		        double sum = t+f;
		        double percentage =  (100/sum) ;
		        std::cout<<"Correct classified: "<<percentage*t<<"%"<<std::endl;

//	End Neural Network

//	std::cout<<result<<std::endl;
//	int right=0;
//	int wrong=0;
//    for(int i =0;i<result.rows;i++)
//    {
////    std::cout<<" Results Predictionnum "<<i<<":  "<<prediction_results.at<float>(i)<<"Results prediction"<<std::endl;
//    	if(i==result.at<float>(i))
//    	{
//    		right++;
//    	}else{
//    		wrong++;
//    	}
//    }
//    std::cout<<"True:"<<right<<"  "<<"False:" << wrong<<std::endl;
//    double sum = right+wrong;
//    double percentage =  (100/sum) ;
//    std::cout<<"Correct classified: "<<percentage*right<<"%"<<std::endl;


//    std::cout<<"Correct "<<std::endl;
}
