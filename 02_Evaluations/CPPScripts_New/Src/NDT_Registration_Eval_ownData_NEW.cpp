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
#include <stdlib.h> 
//#include <cstdlib.h> 
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


int
main()
{

    string ID = "V2";


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
    //ifstream inputFile("Parameters_NDT_Reg.txt");
    //if (!inputFile.is_open()) {
    //    std::cout << "Error opening the file." << std::endl;
    //    return 1;
    //}

    //string line;

    //list<vector<string>> listOfExperiments;
    //while (getline(inputFile, line)) {
        // Use stringstream to split the line into tokens
    //    stringstream ss(line);
    //    string token;
    //    vector<string> tokens;

     //   while (getline(ss, token, ';')) {
     //       tokens.push_back(token);

     //   }

     //   listOfExperiments.push_back(tokens);

    //}


    //inputFile.close();

    //string s = "true";
    //string type;

    //listOfExperiments.pop_front();


    //for (const auto& vec : listOfExperiments) {

    //    //for-loop does not work!!!!!!! Is only executed once

    //    bool_trans = (s == vec[0]);
    //    bool_1D = (s == vec[1]);
    //    bool_2D = (s == vec[2]);
    //    bool_2D_Yaw = (s == vec[3]);
    //    downsample = (s == vec[4]);
    //    ID = vec[5];
    //    ind = stoi(vec[6]);
    //    type = vec[7];
    //    val_diff_tf = stof(vec[8]);
    //    max_step = stof(vec[9]);
    //    voxel_size = stof(vec[10]);
    //    max_iter = stof(vec[11]);

    //    std::cout << "Parameter values successfully read from input file." << std::endl;

    //    if (bool_trans == false)
    //    {
    //        bool_rot = true;
    //    }
    //    else
    //    {
    //        bool_rot = false;
    //    }

        // Lower and upper limits for the random initial pose perturbation
        std::vector<float> lower_limits = { -2,-2,-M_PI / 8};// x, y, yaw
        std::vector<float> upper_limits = { 2,2, M_PI / 8 }; // x, y, yaw

        // How many times should the initial pose be perturbated radnomly?
        int num_runs = 5;
        //bool validation_set = false;


        // Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses(from NDT localization in Autoware) of the Localizer
        string path_map = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\02_Route_1_Data\\FinalMap_Route1.pcd";
        string path_GT_csv = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\05_Data\\Dataset_Validation\\Suburban\\GT_Poses_Suburban_TEMP.csv";
        string path_to_file = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\05_Data\\Dataset_Validation\\Suburban";
        string path_to_scans = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\05_Data\\Dataset_Validation\\Suburban\\prepro_temp\\";


        string name_csv = '\\' + ID + "_NDT_Suburban_TEMP.csv";
        string path_csv;
        path_csv = path_to_file + name_csv;
        string name_csv_iter = '\\' + ID + "_NDT-Iter_Suburban_TEMP.csv";
        string path_csv_iter;
        path_csv_iter = path_to_file + name_csv_iter;


        //Prepare the csv-file containing the transformation matrices and iteration steps
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
        outfile_csv_iter << "ID;Timestamp;Scan Nr.;Run;Init Perturbation (x,y,yaw);Number Iterations;Fitness;t11;t12;t13;t14;t21;t22;t23;t24;t31;t32;t33;t34;t41;t42;t43;t44\n";
        // Close the file
        outfile_csv_iter.close();


        //Prepare the csv-file containing the evaluation metrics
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
        outfile_csv << "ID;Timestamp GT Pose;Scan Nr.;Run;x delta [m];y delta [m];yaw delta [rad];Initial Transl.Error [m];Initial Rot.Error 1 [°];Initial Rot.Error 2 [°];Fitness;Outlier ratio;Transl.Error [m];Rot.Error 1 [°];Rot.Error 2 [°];Number Iterations;Execut.Time[s]\n";
        // Close the file
        outfile_csv.close();

        std::cout << "Prepared the csv files for output.\n" << std::endl;

        //std::vector<string> list_map = {};
        //pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_map(new pcl::PointCloud<pcl::PointXYZ>);
        //std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pointCloudList;

        //if (validation_set == true) {
        //    string path_map_1 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\02_Route_1_Data\\FinalMap_Route1.pcd";
        //    string path_map_2 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\03_Route_2_Data\\FinalMap_Route2.pcd";
        //    string path_map_3 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\04_Route_3_Data\\FinalMap_Route3.pcd";
        //    string path_map_4 = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\05_Route_4_Data\\FinalMap_Route4.pcd";
        //    
        //    list_map = { path_map_1, path_map_2, path_map_3, path_map_4 };

        //    for (int j = 0; j < list_map.size(); j++) {

        //        path_map = list_map[j];
        //        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_map, *pcd_map) == -1)
        //        {
        //            std::cerr << "Failed to load PCD map file: " << path_map << std::endl;
        //            //continue;
        //        }
        //        else {
        //            std::cout << "Loaded " << pcd_map->size() << " data points from map file." << std::endl;
        //            std::cout << "Map path is:" << path_map << std::endl;
        //            pointCloudList.push_back(pcd_map);
        //        }
        //    }
        //}

        // Load GT poses from csv
        rapidcsv::Document doc1(path_GT_csv, rapidcsv::LabelParams(0, -1), rapidcsv::SeparatorParams(';'));
        /*if (validation_set == false) {
            std::vector<float> timestamps = doc1.GetColumn<float>("%time");
        }*/

        std::vector<int> map_idx = doc1.GetColumn<int>("map");
        std::vector<string> pc_scan_path = doc1.GetColumn<string>("pc.timestamp.path");
        std::vector<float> col_x = doc1.GetColumn<float>("field.pose.pose.position.x");
        std::vector<float> col_y = doc1.GetColumn<float>("field.pose.pose.position.y");
        std::vector<float> col_z = doc1.GetColumn<float>("field.pose.pose.position.z");
        std::vector<float> col_alpha = doc1.GetColumn<float>("field.pose.pose.orientation.x");
        std::vector<float> col_beta = doc1.GetColumn<float>("field.pose.pose.orientation.y");
        std::vector<float> col_gamma = doc1.GetColumn<float>("field.pose.pose.orientation.z");
        std::vector<float> col_theta = doc1.GetColumn<float>("field.pose.pose.orientation.w");

        std::vector<float> col_x_delta = doc1.GetColumn<float>("manual.delta.x");
        std::vector<float> col_y_delta = doc1.GetColumn<float>("manual.delta.y");
        std::vector<float> col_z_delta = doc1.GetColumn<float>("manual.delta.z");
        std::vector<float> col_alpha_delta = doc1.GetColumn<float>("manual.delta.orientation.x");
        std::vector<float> col_beta_delta = doc1.GetColumn<float>("manual.delta.orientation.y");
        std::vector<float> col_gamma_delta = doc1.GetColumn<float>("manual.delta.orientation.z");
        std::vector<float> col_theta_delta = doc1.GetColumn<float>("manual.delta.orientation.w");



        std::vector<float> col_x_corrected = {};
        std::vector<float> col_y_corrected = {};
        std::vector<float> col_z_corrected = {};
        std::vector<float> col_alpha_corrected = {};
        std::vector<float> col_beta_corrected = {};
        std::vector<float> col_gamma_corrected = {};
        std::vector<float> col_theta_corrected = {};



       // Correct the GT poses by adding the manual correction deltas
        for (int i = 0; i < pc_scan_path.size(); i++)
        {
            col_x_corrected.push_back(col_x[i] + col_x_delta[i]);
            col_y_corrected.push_back(col_y[i] + col_y_delta[i]);
            col_z_corrected.push_back(col_z[i] + col_z_delta[i]);
            col_alpha_corrected.push_back(col_alpha[i] + col_alpha_delta[i]);
            col_beta_corrected.push_back(col_beta[i] + col_beta_delta[i]);
            col_gamma_corrected.push_back(col_gamma[i] + col_gamma_delta[i]);
            col_theta_corrected.push_back(col_theta[i] + col_theta_delta[i]);
        }
        
        std::cout << "Loaded and prepared the GT positions successfully.\n" << std::endl;

        // Loading the map
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_map(new pcl::PointCloud<pcl::PointXYZ>);
        //
        //if (validation_set == false) {

        //    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_map, *pcd_map) == -1)
        //    {
        //        std::cerr << "Failed to load PCD map file: " << path_map << std::endl;
        //        //continue;
        //    }
        //    else {
        //        std::cout << "Loaded " << pcd_map->size() << " data points from map file for all scans." << std::endl;
        //        std::cout << "Map path is:"<<  path_map << std::endl;
        //    }
        //}

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_map, *pcd_map) == -1)
        {
            std::cerr << "Failed to load PCD map file: " << path_map << std::endl;
            //continue;
        }
        else {
            std::cout << "Loaded " << pcd_map->size() << " data points from map file for all scans." << std::endl;
            std::cout << "Map path is:" << path_map << std::endl;
        }


        //std::string folderPath = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Evaluations\\01_TRC_Skidpad_Data\\Map";
        //std::vector<string> pcdFiles;
        //std::string file;


        //pcl::PointCloud<pcl::PointXYZ>::Ptr combinedCloud(new pcl::PointCloud<pcl::PointXYZ>);

        //for (const auto& entry : fs::directory_iterator(folderPath))
        //{
        //    if (entry.is_regular_file())
        //    {
        //        std::string filename = entry.path().filename().string();
        //        std::string pcdFile = folderPath + "\\" + filename;
        //        //pcdFiles.push_back(entry.path().filename());
        //        // Process the file here
        //        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        //        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFile, *cloud) == -1)
        //        {
        //            std::cerr << "Failed to load PCD file: " << pcdFile << std::endl;
        //            continue;
        //        }

        //        *combinedCloud += *cloud;
        //    }
        //}

        //std::cout << "Loaded " << combinedCloud->size() << " data points from map files." << std::endl;

        //pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::copyPointCloud(*combinedCloud, *target_cloud);

        //std::cout << "Copied map point cloud to target point cloud" << std::endl;


        // For loop starts here
        // For loop does not iterate over all values --> maybe initialize variables outside


        std::string path_pc;
        //int x = 0;
        
        // Iterate through directory containing the preprocessed online scans 

        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);


        std::vector<string> scans= {};
        for (const auto& entry : fs::directory_iterator(path_to_scans))
        {
            scans.push_back(entry.path().string());
        }

        // Sort the file paths to maintain their order
        std::sort(scans.begin(), scans.end());

        std::cout << "The following scans have been found:\n" << std::endl;
        // Print the sorted file paths
        for (const auto& scans : scans) {
            std::cout << scans << std::endl;
        }

        //Initialization
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //Eigen::Quaternionf quaternion_GT(0, 0, 0, 0);
        Eigen::Matrix3f init_rotation_GT;
        Eigen::Vector4f init_translation_GT;
        Eigen::Matrix4f transform_GT;
        Eigen::Vector3f bounds;
        Eigen::Vector4f minPoint;
        Eigen::Vector4f maxPoint;
        pcl::CropBox<pcl::PointXYZ> cropBoxFilter;
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedTargetCloud(new pcl::PointCloud<pcl::PointXYZ>);
        // Initializing Normal Distributions Transform (NDT).
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Matrix4f transform_init;
        std::vector<float> perturbation = { 0,0,0 };
        Eigen::Matrix3f Rot_init;
        Eigen::Vector3f eulerAngles;
        Eigen::Quaternionf quaternion_init;
        Eigen::Matrix4f final_transform;



        // Setting scale dependent NDT parameters
        // Setting minimum transformation difference for termination condition.
        ndt.setTransformationEpsilon(val_diff_tf);
        // Setting maximum step size for More-Thuente line search.
        ndt.setStepSize(max_step);
        //Setting Resolution of NDT grid structure (VoxelGridCovariance).
        ndt.setResolution(voxel_size);

        // Setting max number of registration iterations.
        ndt.setMaximumIterations(max_iter);

        //pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_ptr = empty_cloud;

        //if (validation_set == false) {
        //    // Setting point cloud to be aligned to.
        //    ndt.setInputTarget(pcd_map);
        //}

        ndt.setInputTarget(pcd_map);
        
        int idx_cache = 5;

        for (int x = 0; x < scans.size();x++)
        {
            //if (validation_set == true) {

            //    if (idx_cache != map_idx[x]) {

            //        //path_map = list_map[map_idx[x]];
            //        pcd_map = pointCloudList[map_idx[x]];



            //        //if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_map, *pcd_map) == -1)
            //        //{
            //        //    std::cerr << "Failed to load PCD map file: " << path_map << std::endl;
            //        //    //continue;
            //        //}
            //        //else {
            //        //    std::cout << "Loaded " << pcd_map->size() << " data points from map file." << std::endl;
            //        //    std::cout << "Map path is:" << path_map << std::endl;
            //        //}
            //        // Setting point cloud to be aligned to.
            //        ndt.setInputTarget(pcd_map);
            //        Sleep(3);
            //        std::cout << "Target point cloud successfully set.\n" << std::endl;

            //    }
            //    idx_cache = map_idx[x];
            //} 

            path_pc = scans[x];

            // Loading online scan
            //pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_pc, *input_cloud) == -1)
            {
                PCL_ERROR("Couldn't read file containing online scan \n");
                return (-1);
            }
            
            std::cout << "Loaded " << input_cloud->size() << " data points from online scan file" << std::endl;

            pcl::copyPointCloud(*input_cloud, *source_cloud);

            // Calculate Ground Truth transformation
            Eigen::Quaternionf quaternion_GT(col_theta_corrected[x], col_alpha_corrected[x], col_beta_corrected[x], col_gamma_corrected[x]);
            init_rotation_GT = quaternion_GT.normalized().toRotationMatrix();
            init_translation_GT << col_x_corrected[x], col_y_corrected[x], col_z_corrected[x], 1.0;

            transform_GT.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
            transform_GT.block<3, 3>(0, 0) = init_rotation_GT;
            //transform_GT.block<3, 1>(0, 3) = init_translation_GT;
            transform_GT.col(3) = init_translation_GT;

            std::cout << "Corrected GT position for point cloud Nr.:" << x
                << "\nx: " << col_x_corrected[x] 
                << "\ny: " << col_y_corrected[x]
                << "\nz: " << col_z_corrected[x]
                << "\nquat x: " << col_alpha_corrected[x]
                << "\nquat y: " << col_beta_corrected[x]
                << "\nquat z: " << col_gamma_corrected[x]
                << "\nquat w: " << col_theta_corrected [x] << "\n\n"
                << std::endl;

            // Coping map point cloud
            //pcl::copyPointCloud(*pcd_map, *target_cloud);

            //std::cout << "Copied map point cloud to target point cloud" << std::endl;

            // Cropping PC
            // Define the region of interest (box dimensions)
            //bounds << 200, 200, 100;

            //minPoint << transform_GT(0, 3) - bounds[0], transform_GT(1, 3) - bounds[1], transform_GT(2, 3) - bounds[2], 1.0; // Minimum point coordinates (x, y, z, 1.0)
            //maxPoint << transform_GT(0, 3) + bounds[0], transform_GT(1, 3) + bounds[1], transform_GT(2, 3) + bounds[2], 1.0;// Maximum point coordinates (x, y, z, 1.0)

            // Create the CropBox filter
            //cropBoxFilter.setMin(minPoint);
            //cropBoxFilter.setMax(maxPoint);
            //cropBoxFilter.setNegative(false);  // Set to "true" to crop the region and keep the points outside
            //cropBoxFilter.setInputCloud(target_cloud);

            // Apply the crop filter to obtain the cropped point cloud
            //pcl::copyPointCloud(*target_cloud, *croppedTargetCloud);
            //cropBoxFilter.filter (*croppedTargetCloud); //No cropping because it does not work properly

            // Setting point cloud to be aligned.
            ndt.setInputSource(source_cloud);

            std::cout << "Parameters for NDT algorithm set."
                << std::endl;


            std::srand(static_cast<unsigned>(std::time(nullptr)));

            for (int k = 0; k < num_runs; k++)
            {
                std::cout << "\n#### Run " << k << " ####\n"
                    << std::endl;

                // Add perturbation to the GT pose to get intial pose
                transform_init = transform_GT;

                float range_x = -lower_limits[0] + upper_limits[0];
                float range_y = -lower_limits[1] + upper_limits[1];
                float range_yaw = -lower_limits[2] + upper_limits[2];

                //std::vector<float> perturbation = {(rand() % range_x) + lower_limits[0], (rand() % range_y) + lower_limits[1],(rand() % range_yaw) + lower_limits[2]};
                
                perturbation = { lower_limits[0] + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / range_x)),
                                                    lower_limits[1] + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / range_y)),
                                                    lower_limits[2] + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / range_yaw)) };



                std::cout << "\nPerturbation of initial pose for run Nr.:" << k
                    << "\ndelta x: " << perturbation [0]
                    << "\ndelta y:" << perturbation [1]
                    << "\ndelta yaw: " << perturbation [2] << "\n\n"
                    << std::endl;


                transform_init(0, 3) = transform_init(0, 3) + perturbation[0];
                transform_init(1, 3) = transform_init(1, 3) + perturbation[1];

                Rot_init = transform_init.block<3, 3>(0, 0);
                eulerAngles = Rot_init.eulerAngles(0, 1, 2);

                eulerAngles[2] = eulerAngles[2] + perturbation[2];

                quaternion_init = Eigen::AngleAxisf(eulerAngles[0], Eigen::Vector3f::UnitX()) *
                    Eigen::AngleAxisf(eulerAngles[1], Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(eulerAngles[2], Eigen::Vector3f::UnitZ());

                quaternion_init.normalize();
                Rot_init = quaternion_init.toRotationMatrix();
                transform_init.block<3, 3>(0, 0) = Rot_init;

                //Calculation of the error metrics of the initial perturbated transformation
                auto [rse_transl_init, rso_deg_1_init, rso_deg_2_init] = evalMetrics(transform_init, transform_GT);

                std::cout << "Starting NDT registration now." << std::endl;

                auto start_execution = std::chrono::high_resolution_clock::now();

                ndt.align(*output_cloud, transform_init);

                auto end_execution = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> duration = end_execution - start_execution;
                double outlier_ratio = ndt.getOulierRatio();
                double trans_likelihood = ndt.getTransformationLikelihood();
                double fitness = ndt.getFitnessScore();
                final_transform = ndt.getFinalTransformation();
                int number_iterations = ndt.getFinalNumIteration();

                //Calculation of the error metrics of the proposed transformation
                auto [rse_transl, rso_deg_1, rso_deg_2] = evalMetrics(final_transform, transform_GT);

                std::cout << "\n\nNormal Distributions Transform has converged:" << ndt.hasConverged()
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
                outfile_csv << ID << ";"  << x << ";" << x << ";" << k << ";" << perturbation[0] << ";" << perturbation[1] << ";" << perturbation[2] << ";"
                    << rse_transl_init << ";" << rso_deg_1_init << ";" << rso_deg_2_init << ";" << fitness << ";"
                    << outlier_ratio << ";" << rse_transl << ";" << rso_deg_1 << ";" << rso_deg_2 << ";"
                    << number_iterations << ";" << (duration.count() / 1000) 
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
                outfile_csv_iter << ID << ";" << x << ";" << x << ";" << k << ";" << perturbation[0] << "," << perturbation[1] << "," << perturbation[2] << ";" << number_iterations << ";" << fitness << ";"
                    << final_transform(0, 0) << ";" << final_transform(0, 1) << ";" << final_transform(0, 2) << ";" << final_transform(0, 3) << ";"
                    << final_transform(1, 0) << ";" << final_transform(1, 1) << ";" << final_transform(1, 2) << ";" << final_transform(1, 3) << ";"
                    << final_transform(2, 0) << ";" << final_transform(2, 1) << ";" << final_transform(2, 2) << ";" << final_transform(2, 3) << ";"
                    << final_transform(3, 0) << ";" << final_transform(3, 1) << ";" << final_transform(3, 2) << ";" << final_transform(3, 3) << ";"
                    << std::endl;

                // Close the file
                outfile_csv_iter.close();
                std::cout << "Data written to csv-iter.-file." << std::endl;

            }

            //x = x + 1;
      
 
            std::cout << "\n\nEvaluation completed!" << std::endl;
            input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            source_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            output_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //filtered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //target_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //pcd_map.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //croppedTargetCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            //idx++;
            //Sleep(3);
            //ndt.setInputTarget(empty_cloud);
            std::cout << "PC variables reseted!" << std::endl;
            //std::cout << idx << std::endl;

            continue;
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
//}
    return (0);
}


