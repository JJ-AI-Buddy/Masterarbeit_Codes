// C++ script for evaluation of NDT point cloud registration algorithm
// Influence of pertubations of the initial pose


#include <iostream>
#include <thread>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <pcl/rapidcsv.h>
#include <math.h>
#include <cmath>
#include <pcl/common/io.h>
#include <pcl/filters/crop_box.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <windows.h>

using namespace std::chrono_literals;
using namespace std;
namespace fs = std::filesystem;

pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud;

int findMinIndex(const std::vector<float>& vec) {
    int minIndex = 0;
    int minValue = std::numeric_limits<int>::max();

    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] < minValue) {
            minValue = vec[i];
            minIndex = i;
        }
    }

    return minIndex;
}

auto evalMetrics(Eigen::Matrix4f& tf,Eigen::Matrix4f& tf_GT)
{
    float transl_error = 0.0;
    for (int i = 0; i < 3; i++) {
        transl_error += pow(tf(i,3) - tf_GT(i,3), 2);
    }
    transl_error = sqrt(transl_error);

    //Rodrigues rotation vector for new transformation matrix
    Eigen::Matrix3f rotationMatrixTF = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; i++)
    {
        rotationMatrixTF(i,0) = tf(i,0);
        rotationMatrixTF(i, 1) = tf(i,1);
        rotationMatrixTF(i,2) = tf(i,2);
    }
    Eigen::Vector3f eulerTF = rotationMatrixTF.eulerAngles(0,1,2);
    Eigen::AngleAxisf angleAxisTF (rotationMatrixTF);
    float angleTF = angleAxisTF.angle();
    Eigen::Vector3f axisTF = angleAxisTF.axis();
    axisTF.normalize();
    Eigen::Vector3f rodriguesVectorTF = angleTF * axisTF ;
    rodriguesVectorTF.normalize();

    //Rodrigues rotation vector for GT transformation matrix
    Eigen::Matrix3f rotationMatrixGT = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; i++)
    { 
        rotationMatrixGT(i, 0) = tf_GT(i,0);
        rotationMatrixGT(i, 1) = tf_GT(i,1);
        rotationMatrixGT(i, 2) = tf_GT(i,2);
    }
    Eigen::Vector3f eulerGT = rotationMatrixGT.eulerAngles(0, 1, 2);
    Eigen::AngleAxisf angleAxisGT(rotationMatrixGT);
    float angleGT = angleAxisGT.angle();
    Eigen::Vector3f axisGT = angleAxisGT.axis();
    axisGT.normalize();
    Eigen::Vector3f rodriguesVectorGT = angleGT * axisGT;
    rodriguesVectorGT.normalize();

    float dotProduct = rodriguesVectorGT.dot(rodriguesVectorTF);
    dotProduct = dotProduct > 1.0 ? 1.0 : (dotProduct < -1.0 ? -1.0 : dotProduct);
    float angle = abs(acos(dotProduct));
    float rso_deg_1 = angle * (180.0 / M_PI);

    float rso_deg_2 = abs((eulerGT(2) - eulerTF(2)) * 180.0 / M_PI);

    struct retVals { 
        float x1;
        float x2, x3;
    };

    return retVals { transl_error, rso_deg_1, rso_deg_2 };
}
//float rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS;

std::vector<float> getElementByIndex(const std::list<std::vector<float>>& myList, size_t listIndex)
{
    if (listIndex >= myList.size())
    {
        // Throw an exception or handle the out-of-range list index error
        throw std::out_of_range("List index out of range");
    }

    auto listIt = myList.begin();
    std::advance(listIt, listIndex);

    const std::vector<float>& vectorRef = *listIt;

    //if (vectorIndex >= vectorRef.size())
    //{
        // Throw an exception or handle the out-of-range vector index error
    //    throw std::out_of_range("Vector index out of range");
    //}

    return vectorRef;
}

int index;

int
main()
{
    bool bool_trans = true;
    bool bool_rot;
    bool bool_1D = false;
    bool bool_2D = false;
    bool bool_2D_Yaw = true;

    bool downsample = false;

    string ID = "XXXX";

    if (bool_trans == false)
    {
        bool_rot = true;
    }
    else
    {
        bool_rot = false;
    }

    //Timestamp index
    int ind = 0;

    //Parameters for the NDT algorithm
    // 

    // Setting value minimum transformation difference for termination condition.
    float val_diff_tf = 0.001;
    // Setting maximum step size for More-Thuente line search.
    float max_step = 0.1;
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    float voxel_size = 1.0;
    // Setting max number of registration iterations.
    int max_iter = 30;

    //
    //

    //Read values for all the parameters from txt-file; for automated execution of the evaluation
    ifstream inputFile("Parameters_NDT_Reg.txt");
    if (!inputFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
        return 1;
    }

    string line;

    list<vector<string>> listOfExperiments;
    while (getline(inputFile, line)) {
        // Use stringstream to split the line into tokens
        stringstream ss(line);
        string token;
        vector<string> tokens;

        while (getline(ss, token, ';')) {
            tokens.push_back(token);

        }

        listOfExperiments.push_back(tokens);

    }


    inputFile.close();

    string s = "true";
    string type;

    listOfExperiments.pop_front();


    for (const auto& vec : listOfExperiments) {

        //for-loop does not work!!!!!!! Is only executed once

        bool_trans = (s == vec[0]);
        bool_1D = (s == vec[1]);
        bool_2D = (s == vec[2]);
        bool_2D_Yaw = (s == vec[3]);
        downsample = (s == vec[4]);
        ID = vec[5];
        ind = stoi(vec[6]);
        type = vec[7];
        val_diff_tf = stof(vec[8]);
        max_step = stof(vec[9]);
        voxel_size = stof(vec[10]);
        max_iter = stof(vec[11]);

        std::cout << "Parameter values successfully read from input file." << std::endl;

        if (bool_trans == false)
        {
            bool_rot = true;
        }
        else
        {
            bool_rot = false;
        }


        float lower_limits[6] = { -2,-2,-2,-M_PI / 4,-M_PI / 4, -M_PI / 4 };
        float upper_limits[6] = { 2,2,2,M_PI / 4, M_PI / 4, M_PI / 4 };

        int number_eval_points[6] = { 17,17,17,17,17,17 };    // 6 axes
        int size_eval_points = sizeof(number_eval_points) / sizeof(number_eval_points[0]);

        int axis2Deval[3] = { 0, 1, 5 };
        //

        // Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses(from NDT localization in Autoware) of the Localizer
        string path_map = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\Moriyama_Map.pcd";
        string path_GT_csv = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\14_Local_Pose.csv";
        string path_GNSS_csv = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\13_GNSS_pose.csv";
        string path_to_file = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\03_Moriyama_Evaluation";

        string name_txt = '\\' + ID + type + "BaselineNDTMoriyama.txt";
        string path_txt;
        path_txt = path_to_file + name_txt;
        string name_csv = '\\' + ID + type + "BaselineNDTMoriyama.csv";
        string path_csv;
        path_csv = path_to_file + name_csv;
        string name_csv_iter = '\\' + ID + "_IterStepsBaselineNDTMoriyama.csv";
        string path_csv_iter;
        path_csv_iter = path_to_file + name_csv_iter;


        //Prepare the txt-file
        // 
        // Create an output file stream
        std::ofstream outfile_text;
        // Open the file in output mode
        outfile_text.open(path_txt);

        // Check if the file was opened successfully
        if (!outfile_text) {
            std::cerr << "Error opening txt.-file." << std::endl;
            return 1;
        }

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        // Write data to the file
        outfile_text << "Evaluation of NDT algorithm for map matching on Moriyama dataset \n" << std::endl;
        outfile_text << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << std::endl;
        outfile_text << "\n\n" << std::endl;

        // Close the file
        outfile_text.close();


        //Prepare the csv-file containing the transformation matrices
        // 
        // Create an output file stream
        std::ofstream outfile_csv_iter;
        // Open the file in output mode
        outfile_csv_iter.open(path_csv_iter);

        // Check if the file was opened successfully
        if (!outfile_csv_iter) {
            std::cerr << "Error opening csv.-file." << std::endl;
            return 1;
        }

        // Write data to the file
        outfile_csv_iter << "ID;Timestamp;Axis;Init Error (Trans or Rot);Number Iterations;Fitness;t11;t12;t13;t14;t21;t22;t23;t24;t31;t32;t33;t34;t41;t42;t43;t44\n";

        // Close the file
        outfile_csv_iter.close();

        //Prepare the csv-file
        // 
        // Create an output file stream
        std::ofstream outfile_csv;
        // Open the file in output mode
        outfile_csv.open(path_csv);

        // Check if the file was opened successfully
        if (!outfile_csv) {
            std::cerr << "Error opening csv.-file." << std::endl;
            return 1;
        }

        // Write data to the file
        outfile_csv << "ID; Timestamp GT Pose; Axis; Initial Transl x; Initial Transl y; Initial Transl z; Initial Transl.Error[m]; Initial Rot.Error 1[°]; Initial Rot.Error 2[°]; Fitness; Outlier ratio; Transl.Error[m]; Rot.Error 1[°]; Rot.Error 2[°]; Number Iterations; Execut.Time[s]; GNSS Transl.Error[m]; GNSS Rot.Error 1[°]; GNSS Rot.Error 2[°]\n";

        // Close the file
        outfile_csv.close();

        //read_csv("14_Local_Pose.csv", 3);    // Does not work

        // Load GT poses from csv
        rapidcsv::Document doc1(path_GT_csv);
        std::vector<float> col_time = doc1.GetColumn<float>("%time");
        std::vector<float> col_x = doc1.GetColumn<float>("field.pose.position.x");
        std::vector<float> col_y = doc1.GetColumn<float>("field.pose.position.y");
        std::vector<float> col_z = doc1.GetColumn<float>("field.pose.position.z");
        std::vector<float> col_alpha = doc1.GetColumn<float>("field.pose.orientation.x");
        std::vector<float> col_beta = doc1.GetColumn<float>("field.pose.orientation.y");
        std::vector<float> col_gamma = doc1.GetColumn<float>("field.pose.orientation.z");
        std::vector<float> col_theta = doc1.GetColumn<float>("field.pose.orientation.w");

        int i = 3;
        int sample_step = 100;
        float arr_GT_poses[5][7] = {};

        std::vector<double> timestamps;

        for (int j = 0; j < 5; j++)
        {
            arr_GT_poses[j][0] = col_x[i];
            arr_GT_poses[j][1] = col_y[i];
            arr_GT_poses[j][2] = col_z[i];
            arr_GT_poses[j][3] = col_alpha[i];
            arr_GT_poses[j][4] = col_beta[i];
            arr_GT_poses[j][5] = col_gamma[i];
            arr_GT_poses[j][6] = col_theta[i];

            timestamps.push_back(col_time[i]);

            i = i + sample_step;
        }

        //std::cout << "Test out" << arr_GT_poses[2][3] << "!" << std::endl;

        // Load GNSS poses from csv
        rapidcsv::Document doc2(path_GNSS_csv);
        std::vector<float> col_time_GNSS = doc2.GetColumn<float>("%time");
        std::vector<float> col_x_GNSS = doc2.GetColumn<float>("field.pose.position.x");
        std::vector<float> col_y_GNSS = doc2.GetColumn<float>("field.pose.position.y");
        std::vector<float> col_z_GNSS = doc2.GetColumn<float>("field.pose.position.z");
        std::vector<float> col_alpha_GNSS = doc2.GetColumn<float>("field.pose.orientation.x");
        std::vector<float> col_beta_GNSS = doc2.GetColumn<float>("field.pose.orientation.y");
        std::vector<float> col_gamma_GNSS = doc2.GetColumn<float>("field.pose.orientation.z");
        std::vector<float> col_theta_GNSS = doc2.GetColumn<float>("field.pose.orientation.w");

        i = 0;
        float arr_GNSS_poses[5][7] = {};
        std::vector<float> time_diff;

        for (int j = 0; j < 5; j++)
        {
            for (size_t i = 0; i < col_time_GNSS.size(); ++i)
            {
                time_diff.push_back(col_time_GNSS[i] - timestamps[j]);
            }

            int min_idx = findMinIndex(time_diff);

            arr_GNSS_poses[j][0] = col_x_GNSS[min_idx];
            arr_GNSS_poses[j][1] = col_y_GNSS[min_idx];
            arr_GNSS_poses[j][2] = col_z_GNSS[min_idx];
            arr_GNSS_poses[j][3] = col_alpha_GNSS[min_idx];
            arr_GNSS_poses[j][4] = col_beta_GNSS[min_idx];
            arr_GNSS_poses[j][5] = col_gamma_GNSS[min_idx];
            arr_GNSS_poses[j][6] = col_theta_GNSS[min_idx];
        }

        //std::cout << "Test out" << arr_GNSS_poses[2][3] << "!" << std::endl;

        //Point Clouds
        string path_pc_1 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\PointClouds_Moriyama_140\\1427157790678528.pcd";
        string path_pc_2 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\PointClouds_Moriyama_140\\1427157800687781.pcd";
        string path_pc_3 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\PointClouds_Moriyama_140\\1427157810697012.pcd";
        string path_pc_4 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\PointClouds_Moriyama_140\\1427157820706337.pcd";
        string path_pc_5 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\PointClouds_Moriyama_140\\1427157830815641.pcd";

        string list_path_pc[5] = { path_pc_1, path_pc_2, path_pc_3, path_pc_4, path_pc_5 };

        //std::cout << "Test out" << list_path_pc [3] << "!" << std::endl;

        // Loading the map
        // Read map from several pcd files because only one pcd file is too big
        //

        std::string folderPath = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\pointcloud_map";
        std::vector<string> pcdFiles;
        std::string file;


        pcl::PointCloud<pcl::PointXYZ>::Ptr combinedCloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto& entry : fs::directory_iterator(folderPath))
        {
            if (entry.is_regular_file())
            {
                std::string filename = entry.path().filename().string();
                std::string pcdFile = folderPath + "\\" + filename;
                //pcdFiles.push_back(entry.path().filename());
                // Process the file here
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

                if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFile, *cloud) == -1)
                {
                    std::cerr << "Failed to load PCD file: " << pcdFile << std::endl;
                    continue;
                }

                *combinedCloud += *cloud;
            }
        }

        std::cout << "Loaded " << combinedCloud->size() << " data points from map files." << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*combinedCloud, *target_cloud);

        std::cout << "Copied map point cloud to target point cloud" << std::endl;

        // Open the file in output and append mode
        outfile_text.open(path_txt, std::ios_base::app);

        // Check if the file was opened successfully
        if (!outfile_text) {
            std::cerr << "Error opening txt.-file." << std::endl;
            return 1;
        }

        // Write data to the file
        outfile_text << "\n\nParameter Set:\n" << std::endl;
        outfile_text << "Minimum transformation difference for termination condition: \n" + to_string(val_diff_tf) << std::endl;
        outfile_text << "\nVoxel Size (Resolution of grid structure) in m: \n" + to_string(voxel_size) << std::endl;
        outfile_text << "\nMax. Iterations: \n" + to_string(max_iter) << std::endl;
        outfile_text << "\nMax. Step size for More-Thuente line search: \n" + to_string(max_step) << std::endl;

        // Close the file
        outfile_text.close();
        std::cout << "Data written to txt.-file." << std::endl;

        //float timestamp;

        // For loop starts here
        // For loop does not iterate over all values --> maybe initialize variables outside


        //int upper = 5;
        //ind = 0; 
        //for (ind = 0; ind < upper; ++ind) {   //(sizeof(arr_GT_poses) / sizeof(arr_GT_poses[0]))

        //while (true) {

        //if (m < 5) {
        //float init_pose[7] = { *arr_GT_poses[idx] };
        //float GNSS_pose[7] = { *arr_GNSS_poses[idx] };
        // 
        //double timestamp;
        //timestamp = timestamps[idx];

        //std::cout << "Timestamp " << ind << std::endl;
        //std::cout << "Timestamp " << timestamps[1] << std::endl;
        //std::cout << "Timestamp " << timestamp << std::endl;

        // Loading online scan
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(list_path_pc[ind], *input_cloud) == -1)
        {
            PCL_ERROR("Couldn't read file containing online scan \n");
            return (-1);
        }
        std::cout << "Loaded " << input_cloud->size() << " data points from online scan file" << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*input_cloud, *source_cloud);


        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // Filtering input scan to roughly 10% of original size to increase speed of registration.
        if (downsample == true)
        {
            //pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);
            approximate_voxel_filter.setInputCloud(source_cloud);
            approximate_voxel_filter.filter(*filtered_cloud);
            std::cout << "Filtered cloud contains " << filtered_cloud->size()
                << " data points from room_scan2.pcd" << std::endl;
        }
        else
        {
            //pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*source_cloud, *filtered_cloud);
            //filtered_cloud = source_cloud;
            std::cout << "Point cloud not filtered. " << std::endl;
        }

        //std::cout << "TEST_1 " << std::to_string(arr_GT_poses[ind][0]) << std::endl;

        // Calculate Ground Truth transformation
        Eigen::Quaternionf quaternion_GT(arr_GT_poses[ind][6], arr_GT_poses[ind][3], arr_GT_poses[ind][4], arr_GT_poses[ind][5]);
        Eigen::Matrix3f init_rotation_GT = quaternion_GT.normalized().toRotationMatrix();
        Eigen::Vector4f init_translation_GT; init_translation_GT << arr_GT_poses[ind][0], arr_GT_poses[ind][1], arr_GT_poses[ind][2], 1.0;
        Eigen::Matrix4f transform_GT;
        transform_GT.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
        transform_GT.block<3, 3>(0, 0) = init_rotation_GT;
        //transform_GT.block<3, 1>(0, 3) = init_translation_GT;
        transform_GT.col(3) = init_translation_GT;

        // = (init_translation_GT * init_rotation_GT).matrix();
        //Eigen::Matrix4f* transform_GT_pointer = &transform_GT;

        //Calculate GNSS transformation
        Eigen::Quaternionf quaternion_GNSS(arr_GNSS_poses[ind][6], arr_GNSS_poses[ind][3], arr_GNSS_poses[ind][4], arr_GNSS_poses[ind][5]);
        Eigen::Matrix3f init_rotation_GNSS = quaternion_GNSS.normalized().toRotationMatrix();
        Eigen::Vector4f init_translation_GNSS; init_translation_GNSS << arr_GNSS_poses[ind][0], arr_GNSS_poses[ind][1], arr_GNSS_poses[ind][2], 1.0;

        Eigen::Matrix4f transform_GNSS;
        transform_GNSS.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
        transform_GNSS.block<3, 3>(0, 0) = init_rotation_GNSS;
        //transform_GNSS.block<3, 1>(0, 3) = init_translation_GNSS;
        transform_GNSS.col(3) = init_translation_GNSS;


        //Eigen::Matrix4f transform_GNSS = (init_translation_GNSS * init_rotation_GNSS).matrix();
        //Eigen::Matrix4f* transform_GNSS_pointer = &transform_GNSS;

        //std::cout << "TEST_1 " << transform_GNSS(0, 3)
        //    << " TEST_2 " << transform_GT(1, 3)
            //   << " TEST_3 " << transform_GT(2, 2) << std::endl;

        auto [rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS] = evalMetrics(transform_GNSS, transform_GT);

        //std::cout << "TEST_1 " << rse_transl_GNSS
        //    << " TEST_2 " << rso_deg_1_GNSS
        //    << " TEST_3 " << rso_deg_2_GNSS << std::endl;

        // Cropping PC
        // Define the region of interest (box dimensions)
        Eigen::Vector3f bounds(200, 200, 100);

        Eigen::Vector4f minPoint; minPoint << transform_GT(0, 3) - bounds[0], transform_GT(1, 3) - bounds[1], transform_GT(2, 3) - bounds[2], 1.0; // Minimum point coordinates (x, y, z, 1.0)
        Eigen::Vector4f maxPoint; maxPoint << transform_GT(0, 3) + bounds[0], transform_GT(1, 3) + bounds[1], transform_GT(2, 3) + bounds[2], 1.0;// Maximum point coordinates (x, y, z, 1.0)

        // Create the CropBox filter
        pcl::CropBox<pcl::PointXYZ> cropBoxFilter;
        cropBoxFilter.setMin(minPoint);
        cropBoxFilter.setMax(maxPoint);
        cropBoxFilter.setNegative(false);  // Set to "true" to crop the region and keep the points outside
        cropBoxFilter.setInputCloud(target_cloud);

        // Apply the crop filter to obtain the cropped point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedTargetCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*target_cloud, *croppedTargetCloud);
        //cropBoxFilter.filter (*croppedTargetCloud);

        std::cout << "Cropped target pc contains now " << croppedTargetCloud->size() << " points." << std::endl;

        // Write to txt.-File
        // 

        // Open the file in output and append mode
        outfile_text.open(path_txt, std::ios_base::app);

        // Check if the file was opened successfully
        if (!outfile_text) {
            std::cerr << "Error opening txt.-file." << std::endl;
            return 1;
        }

        // Write data to the file
        outfile_text << "\n\n" << "-------------------------------------------" << std::endl;
        outfile_text << "\n\nSource point cloud:\nTimestamp = " + to_string(timestamps[0]) << std::endl;
        outfile_text << "\nNumber of points = " + to_string(filtered_cloud->size()) << std::endl;
        outfile_text << "Cropped (Y/N)? = N\n\n" << std::endl;
        outfile_text << "Target point cloud: Moriyama map,\nNumber of points after cropping (RoI):" + to_string(croppedTargetCloud->size()) << std::endl;
        outfile_text << "Cropped (Y/N)? = Y\n\n" << std::endl;
        outfile_text << "GT Transformation matrix:\n" << transform_GT << std::endl;

        // Close the file
        outfile_text.close();
        std::cout << "Data written to txt.-file." << std::endl;


        list < vector<float>> list_eval_points;
        double start;
        double end;
        double step;
        int numSamples;

        for (int i = 0; i < size_eval_points; i++) {
            std::vector<float> result_linspace(number_eval_points[i], 0.0);
            start = lower_limits[i];
            end = upper_limits[i];
            numSamples = number_eval_points[i];

            step = (end - start) / (numSamples - 1);

            for (int j = 0; j < numSamples; j++) {
                result_linspace[j] = start + step * j;
            }
            list_eval_points.push_back(result_linspace);
        }

        // Initializing Normal Distributions Transform (NDT).
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

        // Setting scale dependent NDT parameters
        // Setting minimum transformation difference for termination condition.
        ndt.setTransformationEpsilon(val_diff_tf);
        // Setting maximum step size for More-Thuente line search.
        ndt.setStepSize(max_step);
        //Setting Resolution of NDT grid structure (VoxelGridCovariance).
        ndt.setResolution(voxel_size);

        // Setting max number of registration iterations.
        ndt.setMaximumIterations(max_iter);

        // Setting point cloud to be aligned.
        ndt.setInputSource(filtered_cloud);
        // Setting point cloud to be aligned to.
        ndt.setInputTarget(croppedTargetCloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        int counter;

        if (bool_1D == true) {

            std::cout << "\nStarting NDT algorithm evaluation.\nMethod: Pertubate initial pose guess in only one direction (Translatory x,y,z or rotatory).\nTimestamp: "
                << timestamps[ind] << " with ID: " << ind << " \n" << std::endl;

            for (int k = 0; k < 3; k++) {

                if (bool_trans == true) {
                    counter = k;
                }

                if (bool_rot == true) {
                    counter = k + 3;
                }


                for (int n = 0; n < number_eval_points[counter]; n++) {

                    Eigen::Matrix4f transform_init = transform_GT;
                    std::vector<float> eval_points = getElementByIndex(list_eval_points, counter);

                    std::vector<float> transl(3, 0.0);
                    std::vector<float> rotl(3, 0.0);
                    std::vector<float> output_offset(3, 0.0);

                    if (bool_trans == true) {

                        std::cout << "\nTranslatory: NDT Registration for axis ID " << counter << " with initial offset " << eval_points[n] << ".\n" << std::endl;

                        //Only translatory perturbation
                        transl[k] = eval_points[n];
                        transform_init(0, 3) = transform_init(0, 3) + transl[0];
                        transform_init(1, 3) = transform_init(1, 3) + transl[1];
                        transform_init(2, 3) = transform_init(2, 3) + transl[2];

                        output_offset = transl;
                    }

                    if (bool_rot == true) {

                        std::cout << "\nRotatory: NDT Registration for axis ID " << counter << " with initial offset " << eval_points[n] << ".\n" << std::endl;


                        //Only rotatory perturbation

                        rotl[k] = eval_points[n];

                        Eigen::Matrix3f Rot_init = transform_init.block<3, 3>(0, 0);
                        Eigen::Vector3f eulerAngles = Rot_init.eulerAngles(0, 1, 2);

                        for (int w = 0; w < 3; w++) {
                            eulerAngles[w] = eulerAngles[w] + rotl[w];
                        }

                        Eigen::Quaternionf quaternion_init;
                        quaternion_init = Eigen::AngleAxisf(eulerAngles[0], Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(eulerAngles[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(eulerAngles[2], Eigen::Vector3f::UnitZ());

                        quaternion_init.normalize();
                        Rot_init = quaternion_init.toRotationMatrix();
                        transform_init.block<3, 3>(0, 0) = Rot_init;

                        output_offset = rotl;

                    }
                    //Calculation of the error metrics of the initial perturbated transformation
                    auto [rse_transl_init, rso_deg_1_init, rso_deg_2_init] = evalMetrics(transform_init, transform_GT);

                    auto start_execution = std::chrono::high_resolution_clock::now();

                    // Calculating required rigid transform to align the input cloud to the target cloud.
                    //pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    ndt.align(*output_cloud, transform_init);

                    auto end_execution = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> duration = end_execution - start_execution;
                    double outlier_ratio = ndt.getOulierRatio();
                    double trans_likelihood = ndt.getTransformationLikelihood();
                    double fitness = ndt.getFitnessScore();
                    Eigen::Matrix4f final_transform = ndt.getFinalTransformation();
                    int number_iterations = ndt.getFinalNumIteration();

                    //Calculation of the error metrics of the proposed transformation
                    auto [rse_transl, rso_deg_1, rso_deg_2] = evalMetrics(final_transform, transform_GT);


                    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
                        << "\nAfter " << number_iterations << " Iteration(s)."
                        << "\nThe registration took " << (duration.count() / 1000) << " seconds to converge."
                        << "\nFitness score: " << fitness
                        << "\nOutlier ratio: " << outlier_ratio
                        << "\nTransformation likelihood: " << trans_likelihood
                        << "\nTranslation error: " << rse_transl
                        << "\nRelative rotation error 1: " << rso_deg_1
                        << "\nRelative rotation error 2: " << rso_deg_2
                        << std::endl;

                    // Open the csv-file in output mode
                    outfile_csv.open(path_csv, std::ios_base::app);

                    // Check if the file was opened successfully
                    if (!outfile_csv) {
                        std::cerr << "Error opening csv.-file." << std::endl;
                        return 1;
                    }

                    // Write data to the file
                    outfile_csv << ID << ";" << timestamps[ind] << ";" << k << ";" << output_offset[0] << ";" << output_offset[1] << ";" << output_offset[2] << ";"
                        << rse_transl_init << ";" << rso_deg_1_init << ";" << rso_deg_2_init << ";" << fitness << ";"
                        << outlier_ratio << ";" << rse_transl << ";" << rso_deg_1 << ";" << rso_deg_2 << ";"
                        << number_iterations << ";" << (duration.count() / 1000) << ";" << rse_transl_GNSS << ";" << rso_deg_1_GNSS << ";" << rso_deg_2_GNSS
                        << std::endl;


                    // Close the file
                    outfile_csv.close();
                    std::cout << "Data written to csv.-file." << std::endl;



                    // Open the csv-iter-file in output mode
                    outfile_csv_iter.open(path_csv_iter, std::ios_base::app);

                    // Check if the file was opened successfully
                    if (!outfile_csv_iter) {
                        std::cerr << "Error opening csv.-file." << std::endl;
                        return 1;
                    }

                    // Write data to the file
                    outfile_csv_iter << ID << ";" << timestamps[ind] << ";" << k << ";" << eval_points[n] << ";" << number_iterations << ";" << fitness << ";"
                        << final_transform(0, 0) << ";" << final_transform(0, 1) << ";" << final_transform(0, 2) << ";" << final_transform(0, 3) << ";"
                        << final_transform(1, 0) << ";" << final_transform(1, 1) << ";" << final_transform(1, 2) << ";" << final_transform(1, 3) << ";"
                        << final_transform(2, 0) << ";" << final_transform(2, 1) << ";" << final_transform(2, 2) << ";" << final_transform(2, 3) << ";"
                        << final_transform(3, 0) << ";" << final_transform(3, 1) << ";" << final_transform(3, 2) << ";" << final_transform(3, 3) << ";"
                        << std::endl;

                    // Close the file
                    outfile_csv_iter.close();
                    std::cout << "Data written to csv-iter.-file." << std::endl;


                    
                }
            }
            std::cout << "\n\nEvaluation completed!" << std::endl;
            input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            source_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            output_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            filtered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            target_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            croppedTargetCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //idx++;
            Sleep(3);
            std::cout << "PC variables reseted!" << std::endl;
            //std::cout << idx << std::endl;
        }

        if (bool_2D == true) {

            std::cout << "\nStarting NDT algorithm evaluation.\nMethod: Pertubate initial pose guess in two directions (preferably x and y).\nTimestamp: "
                << timestamps[ind] << " with ID: " << ind << " \n" << std::endl;

            for (int k = 0; k < number_eval_points[axis2Deval[0]]; k++) {
                //Iteratve over all evaluation points of first axis

                for (int n = 0; n < number_eval_points[axis2Deval[1]]; n++) {
                    //Iterate over all evaluation points of second axis

                    std::vector<float> list_axes(3, 0.0);

                    list_axes[axis2Deval[0]] = getElementByIndex(list_eval_points, axis2Deval[0])[k];
                    list_axes[axis2Deval[1]] = getElementByIndex(list_eval_points, axis2Deval[1])[n];

                    Eigen::Matrix4f transform_init = transform_GT;


                    std::cout << "\nNDT Registration for axis_1 ID " << axis2Deval[0] << " with initial offset " << list_axes[0]
                        << " and axis_2 ID " << axis2Deval[1] << " with initial offset " << list_axes[1] << ".\n" << std::endl;


                    if (bool_trans == true) {
                        //Translatory perturbation
                        transform_init(0, 3) = transform_init(0, 3) + list_axes[0];
                        transform_init(1, 3) = transform_init(1, 3) + list_axes[1];
                        transform_init(2, 3) = transform_init(2, 3) + list_axes[2];

                    }

                    if (bool_rot == true) {
                        //Rotatory perturbation
                        Eigen::Matrix3f Rot_init = transform_init.block<3, 3>(0, 0);
                        Eigen::Vector3f eulerAngles = Rot_init.eulerAngles(0, 1, 2);

                        for (int w = 0; w < 3; w++) {
                            eulerAngles[w] = eulerAngles[w] + list_axes[w];
                        }

                        Eigen::Quaternionf quaternion_init;
                        quaternion_init = Eigen::AngleAxisf(eulerAngles[0], Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(eulerAngles[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(eulerAngles[2], Eigen::Vector3f::UnitZ());

                        quaternion_init.normalize();
                        Rot_init = quaternion_init.toRotationMatrix();
                        transform_init.block<3, 3>(0, 0) = Rot_init;

                    }

                    //Calculation of error metrics of the initial perturbated transformation
                    auto [rse_transl_init, rso_deg_1_init, rso_deg_2_init] = evalMetrics(transform_init, transform_GT);

                    auto start_execution = std::chrono::high_resolution_clock::now();

                    // Calculating required rigid transform to align the input cloud to the target cloud.
                    //pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    ndt.align(*output_cloud, transform_init);

                    auto end_execution = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> duration = end_execution - start_execution;
                    double outlier_ratio = ndt.getOulierRatio();
                    double trans_likelihood = ndt.getTransformationLikelihood();
                    double fitness = ndt.getFitnessScore();
                    Eigen::Matrix4f final_transform = ndt.getFinalTransformation();
                    int number_iterations = ndt.getFinalNumIteration();

                    //Calculation of error metrics of the proposed transformation
                    auto [rse_transl, rso_deg_1, rso_deg_2] = evalMetrics(final_transform, transform_GT);


                    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
                        << "\nAfter " << number_iterations << " Iteration(s)."
                        << "\nThe registration took " << (duration.count() / 1000) << " seconds to converge."
                        << "\nFitness score: " << fitness
                        << "\nOutlier ratio: " << outlier_ratio
                        << "\nTransformation likelihood: " << trans_likelihood
                        << "\nTranslation error: " << rse_transl
                        << "\nRelative rotation error 1: " << rso_deg_1
                        << "\nRelative rotation error 2: " << rso_deg_2
                        << std::endl;

                    // Open the csv-file in output mode
                    outfile_csv.open(path_csv, std::ios_base::app);

                    // Check if the file was opened successfully
                    if (!outfile_csv) {
                        std::cerr << "Error opening csv.-file." << std::endl;
                        return 1;
                    }

                    // Write data to the file
                    outfile_csv << ID << ";" << timestamps[ind] << ";" << axis2Deval[0] << "," << axis2Deval[1] << ";" << list_axes[0] << ";" << list_axes[1] << ";" << list_axes[2] << ";"
                        << rse_transl_init << ";" << rso_deg_1_init << ";" << rso_deg_2_init << ";" << fitness << ";"
                        << outlier_ratio << ";" << rse_transl << ";" << rso_deg_1 << ";" << rso_deg_2 << ";"
                        << number_iterations << ";" << (duration.count() / 1000) << ";" << rse_transl_GNSS << ";" << rso_deg_1_GNSS << ";" << rso_deg_2_GNSS
                        << std::endl;


                    // Close the file
                    outfile_csv.close();
                    std::cout << "Data written to csv.-file." << std::endl;



                    // Open the csv-iter-file in output mode
                    outfile_csv_iter.open(path_csv_iter, std::ios_base::app);

                    // Check if the file was opened successfully
                    if (!outfile_csv_iter) {
                        std::cerr << "Error opening csv.-file." << std::endl;
                        return 1;
                    }

                    // Write data to the file
                    outfile_csv_iter << ID << ";" << timestamps[ind] << ";" << axis2Deval[0] << "," << axis2Deval[1]
                        << ";" << rse_transl_init << ";" << number_iterations << ";" << fitness << ";"
                        << final_transform(0, 0) << ";" << final_transform(0, 1) << ";" << final_transform(0, 2) << ";" << final_transform(0, 3) << ";"
                        << final_transform(1, 0) << ";" << final_transform(1, 1) << ";" << final_transform(1, 2) << ";" << final_transform(1, 3) << ";"
                        << final_transform(2, 0) << ";" << final_transform(2, 1) << ";" << final_transform(2, 2) << ";" << final_transform(2, 3) << ";"
                        << final_transform(3, 0) << ";" << final_transform(3, 1) << ";" << final_transform(3, 2) << ";" << final_transform(3, 3) << ";"
                        << std::endl;

                    // Close the file
                    outfile_csv_iter.close();
                    std::cout << "Data written to csv-iter.-file." << std::endl;

                }
            }
            std::cout << "\n\nEvaluation completed!" << std::endl;
            input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            source_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            output_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            filtered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            target_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            croppedTargetCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //idx++;
            Sleep(3);
            std::cout << "PC variables reseted!" << std::endl;
            //std::cout << idx << std::endl;
        }


        if (bool_2D_Yaw == true) {

            std::cout << "\nStarting NDT algorithm evaluation.\nMethod: Pertubate initial pose guess in x and y direction + yaw angle (z-axis) offset.\nTimestamp: "
                << timestamps[ind] << " with ID: " << ind << " \n" << std::endl;


            for (int k = 0; k < number_eval_points[axis2Deval[0]]; k++) {
                //Iterate over all evaluation points of the x-axis

                for (int n = 0; n < number_eval_points[axis2Deval[1]]; n++) {
                    //Iterate over all evaluation points of the y-axis

                    for (int w = 0; w < number_eval_points[axis2Deval[2]]; w++) {
                        //Iterate over all evaluation points of the yaw angle (around z-axis)

                        std::vector<float> list_axes(3, 0.0);

                        list_axes[0] = getElementByIndex(list_eval_points, axis2Deval[0])[k];
                        list_axes[1] = getElementByIndex(list_eval_points, axis2Deval[1])[n];
                        list_axes[2] = getElementByIndex(list_eval_points, axis2Deval[2])[w];

                        std::cout << "\nNDT Registration for axis_1 ID " << axis2Deval[0] << " with initial offset " << list_axes[0]
                            << " and axis_2 ID " << axis2Deval[1] << " with initial offset " << list_axes[1] <<
                            " and yaw angle (z-axis) offset " << list_axes[2] << ".\n" << std::endl;

                        //Translatory pertubation of the initial transformation matrix (here only in x and y direction)
                        Eigen::Matrix4f transform_init = transform_GT;

                        transform_init(0, 3) = transform_init(0, 3) + list_axes[0];
                        transform_init(1, 3) = transform_init(1, 3) + list_axes[1];

                        //Rotatory pertubation of the initial transformation matrix (here only around z-axis)
                        Eigen::Matrix3f Rot_init = transform_init.block<3, 3>(0, 0);
                        Eigen::Vector3f eulerAngles = Rot_init.eulerAngles(0, 1, 2);

                        eulerAngles[2] = eulerAngles[2] + list_axes[2];

                        Eigen::Quaternionf quaternion_init;
                        quaternion_init = Eigen::AngleAxisf(eulerAngles[0], Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(eulerAngles[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(eulerAngles[2], Eigen::Vector3f::UnitZ());

                        quaternion_init.normalize();
                        Rot_init = quaternion_init.toRotationMatrix();
                        transform_init.block<3, 3>(0, 0) = Rot_init;


                        //Calculation of error metrics of the initial perturbated transformation
                        auto [rse_transl_init, rso_deg_1_init, rso_deg_2_init] = evalMetrics(transform_init, transform_GT);

                        auto start_execution = std::chrono::high_resolution_clock::now();

                        // Calculating required rigid transform to align the input cloud to the target cloud.
                        //pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                        ndt.align(*output_cloud, transform_init);

                        auto end_execution = std::chrono::high_resolution_clock::now();

                        std::chrono::duration<double, std::milli> duration = end_execution - start_execution;
                        double outlier_ratio = ndt.getOulierRatio();
                        double trans_likelihood = ndt.getTransformationLikelihood();
                        double fitness = ndt.getFitnessScore();
                        Eigen::Matrix4f final_transform = ndt.getFinalTransformation();
                        int number_iterations = ndt.getFinalNumIteration();

                        //Calculation of error metrics of the proposed transformation
                        auto [rse_transl, rso_deg_1, rso_deg_2] = evalMetrics(final_transform, transform_GT);


                        std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
                            << "\nAfter " << number_iterations << " Iteration(s)."
                            << "\nThe registration took " << (duration.count() / 1000) << " seconds to converge."
                            << "\nFitness score: " << fitness
                            << "\nOutlier ratio: " << outlier_ratio
                            << "\nTransformation likelihood: " << trans_likelihood
                            << "\nTranslation error: " << rse_transl
                            << "\nRelative rotation error 1: " << rso_deg_1
                            << "\nRelative rotation error 2: " << rso_deg_2
                            << std::endl;

                        // Open the csv-file in output mode
                        outfile_csv.open(path_csv, std::ios_base::app);

                        // Check if the file was opened successfully
                        if (!outfile_csv) {
                            std::cerr << "Error opening csv.-file." << std::endl;
                            return 1;
                        }

                        // Write data to the file
                        outfile_csv << ID << ";" << timestamps[ind] << ";" << axis2Deval[0] << "," << axis2Deval[1] << "," << axis2Deval[2]
                            << ";" << list_axes[0] << ";" << list_axes[1] << ";" << list_axes[2] << ";"
                            << rse_transl_init << ";" << rso_deg_1_init << ";" << rso_deg_2_init << ";" << fitness << ";"
                            << outlier_ratio << ";" << rse_transl << ";" << rso_deg_1 << ";" << rso_deg_2 << ";"
                            << number_iterations << ";" << (duration.count() / 1000) << ";" << rse_transl_GNSS << ";" << rso_deg_1_GNSS << ";" << rso_deg_2_GNSS
                            << std::endl;


                        // Close the file
                        outfile_csv.close();
                        std::cout << "Data written to csv.-file." << std::endl;



                        // Open the csv-iter-file in output mode
                        outfile_csv_iter.open(path_csv_iter, std::ios_base::app);

                        // Check if the file was opened successfully
                        if (!outfile_csv_iter) {
                            std::cerr << "Error opening csv.-file." << std::endl;
                            return 1;
                        }

                        // Write data to the file
                        outfile_csv_iter << ID << ";" << timestamps[ind] << ";" << axis2Deval[0] << "," << axis2Deval[1] << "," << axis2Deval[2] << ";"
                            << rse_transl_init << ";" << number_iterations << ";" << fitness << ";"
                            << final_transform(0, 0) << ";" << final_transform(0, 1) << ";" << final_transform(0, 2) << ";" << final_transform(0, 3) << ";"
                            << final_transform(1, 0) << ";" << final_transform(1, 1) << ";" << final_transform(1, 2) << ";" << final_transform(1, 3) << ";"
                            << final_transform(2, 0) << ";" << final_transform(2, 1) << ";" << final_transform(2, 2) << ";" << final_transform(2, 3) << ";"
                            << final_transform(3, 0) << ";" << final_transform(3, 1) << ";" << final_transform(3, 2) << ";" << final_transform(3, 3) << ";"
                            << std::endl;

                        // Close the file
                        outfile_csv_iter.close();
                        std::cout << "Data written to csv-iter.-file." << std::endl;


                        //std::cout << "\n\nEvaluation completed!" << std::endl;
                        input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        source_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        output_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        filtered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        target_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        croppedTargetCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                        //idx++;
                        Sleep(3);
                        std::cout << "PC variables reseted!" << std::endl;
                        //std::cout << idx << std::endl;
                    }
                }

#
            }
            std::cout << "\n\nEvaluation completed!" << std::endl;
            input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            source_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            output_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            filtered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            target_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            croppedTargetCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //idx++;
            Sleep(3);
            std::cout << "PC variables reseted!" << std::endl;
            //std::cout << idx << std::endl;
        }
       

        //std::cout << idx << std::endl;


        // Transforming unfiltered, input cloud using found transform.
        //pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());

        // Saving transformed input cloud.
        //pcl::io::savePCDFileASCII("room_scan2_transformed.pcd", *output_cloud);

        // Initializing point cloud visualizer
        //pcl::visualization::PCLVisualizer::Ptr
        //viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
        //viewer_final->setBackgroundColor(0, 0, 0);

        // Coloring and visualizing target cloud (red).
        //pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ>
        //target_color(croppedTargetCloud, 255, 0, 0);
        //viewer_final->addPointCloud<pcl::PointXYZ>(croppedTargetCloud, target_color, "target cloud");
        //viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        //                                              1, "target cloud");

        // Coloring and visualizing transformed input cloud (green).
        //pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ>
        //output_color(output_cloud, 0, 255, 0);
        //viewer_final->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output cloud");
        //viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        //                                              1, "output cloud");

        // Starting visualizer
        //viewer_final->addCoordinateSystem(1.0, "global");
        //viewer_final->initCameraParameters();

        // Wait until visualizer window is closed.
        //while (!viewer_final->wasStopped())
        //{
        //    viewer_final->spinOnce(100);
        //    std::this_thread::sleep_for(100ms);
        //}
    }
    return (0);
}


