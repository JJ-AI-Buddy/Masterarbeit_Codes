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
    Eigen::Vector3f rodriguesVectorTF = angleTF * axisTF;

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

    float dotProduct = rodriguesVectorTF.dot(rodriguesVectorGT);
    float angle = abs(acos(dotProduct));
    float rso_deg_1 = angle * 180.0 / M_PI;

    float rso_deg_2 = abs((eulerGT(2) - eulerTF(2)) * 180.0 / M_PI);

    struct retVals { 
        float x1;
        float x2, x3;
    };

    return retVals { transl_error, rso_deg_1, rso_deg_2 };
}
//float rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS;

int
main()
{
    bool bool_trans = true;
    bool bool_rot;
    bool bool_1D = false;
    bool bool_2D = false;
    bool bool_2D_Yaw = false;

    bool downsample = false;

    string ID = "C200";

    if (bool_trans == false)
    {
        bool_rot = true;
    }
    else
    {
        bool_rot = false;
    }

    //Parameters for the NDT algorithm
    // 

    // Setting value minimum transformation difference for termination condition.
    float val_diff_tf = 0.1;
    // Setting maximum step size for More-Thuente line search.
    float max_step = 0.1;
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    float voxel_size = 1.0;
    // Setting max number of registration iterations.
    int max_iter = 30;

    //
    //
    float lower_limits[6] = { -2,-2,-2,-M_PI / 4,-M_PI / 4, -M_PI / 4 };
    float upper_limits[6] = { 2,2,2,M_PI / 4, M_PI / 4, M_PI / 4 };

    int number_eval_points[6] = { 17,17,17,17,17,17 };    // 6 axes

    int axis2Deval[3] = { 0, 1, 5 };
    //
    //


    // Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses(from NDT localization in Autoware) of the Localizer
    string path_map = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\Moriyama_Map.pcd";
    string path_GT_csv = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\14_Local_Pose.csv";
    string path_GNSS_csv = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\02_Moriyama_Data\\13_GNSS_pose.csv";
    string path_to_file = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Baseline\\03_Moriyama_Evaluation";

    string name_txt = '\\' + ID + "_TranslBaselineNDTMoriyama.csv";
    string path_txt;
    path_txt = path_to_file + name_txt;
    string name_csv = '\\' + ID + "_TranslBaselineNDTMoriyama.csv";
    string path_csv;
    path_csv = path_to_file + name_csv;
    string name_csv_iter = '\\' + ID + "_IterStepsBaselineNDTMoriyama.csv";
    string path_csv_iter;
    path_csv_iter = path_to_file + name_csv_iter;

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

    std::vector<float> timestamps;

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


    // For loop starts here
    int idx = 0;
    //float init_pose[7] = { *arr_GT_poses[0] };
    //float GNSS_pose[7] = { *arr_GNSS_poses[0] };
    //float timestamp = timestamps[0];


    // Loading online scan
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(list_path_pc[0], *input_cloud) == -1)
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

    std::cout << "TEST_1 " << std::to_string(arr_GT_poses[idx][0]) << std::endl;

    // Works until here - 07.07.2023

    // Calculate Ground Truth transformation
    Eigen::Quaternionf quaternion_GT (arr_GT_poses[idx][6], arr_GT_poses[idx][3], arr_GT_poses[idx][4], arr_GT_poses[idx][5]);
    Eigen::Matrix3f init_rotation_GT = quaternion_GT.normalized().toRotationMatrix();
    Eigen::Vector4f init_translation_GT; init_translation_GT << arr_GT_poses[idx][0], arr_GT_poses[idx][1], arr_GT_poses[idx][2], 1.0;
    Eigen::Matrix4f transform_GT;
    transform_GT.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
    transform_GT.block<3, 3>(0, 0) = init_rotation_GT;
    //transform_GT.block<3, 1>(0, 3) = init_translation_GT;
    transform_GT.col(3) = init_translation_GT;

     // = (init_translation_GT * init_rotation_GT).matrix();
    //Eigen::Matrix4f* transform_GT_pointer = &transform_GT;

    //Calculate GNSS transformation
    Eigen::Quaternionf quaternion_GNSS (arr_GNSS_poses [idx][6], arr_GNSS_poses[idx][3], arr_GNSS_poses[idx][4], arr_GNSS_poses[idx][5]);
    Eigen::Matrix3f init_rotation_GNSS = quaternion_GNSS.normalized().toRotationMatrix();
    Eigen::Vector4f init_translation_GNSS; init_translation_GNSS << arr_GNSS_poses[idx][0], arr_GNSS_poses[idx][1], arr_GNSS_poses[idx][2], 1.0;

    Eigen::Matrix4f transform_GNSS;
    transform_GNSS.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
    transform_GNSS.block<3, 3>(0, 0) = init_rotation_GNSS;
    //transform_GNSS.block<3, 1>(0, 3) = init_translation_GNSS;
    transform_GNSS.col(3) = init_translation_GNSS;


    //Eigen::Matrix4f transform_GNSS = (init_translation_GNSS * init_rotation_GNSS).matrix();
    //Eigen::Matrix4f* transform_GNSS_pointer = &transform_GNSS;

    std::cout << "TEST_1 " << transform_GNSS (0,3)
        << " TEST_2 " << transform_GT (1,3)
        << " TEST_3 " << transform_GT(2,2) << std::endl;

    auto [rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS] = evalMetrics(transform_GNSS, transform_GT);

    std::cout << "TEST_1 " << rse_transl_GNSS
        << " TEST_2 " << rso_deg_1_GNSS
        << " TEST_3 " << rso_deg_2_GNSS << std::endl;

    // Cropping PC
    // Define the region of interest (box dimensions)
    Eigen::Vector3f bounds(200, 200, 100);

    Eigen::Vector4f minPoint; minPoint << transform_GT(0,3) - bounds[0], transform_GT(1, 3) - bounds[1], transform_GT(2, 3) - bounds[2], 1.0; // Minimum point coordinates (x, y, z, 1.0)
    Eigen::Vector4f maxPoint; maxPoint << transform_GT(0, 3) + bounds[0], transform_GT(1, 3) + bounds[1], transform_GT(2, 3) + bounds[2], 1.0;// Maximum point coordinates (x, y, z, 1.0)

    // Create the CropBox filter
    pcl::CropBox<pcl::PointXYZ> cropBoxFilter;
    cropBoxFilter.setInputCloud(target_cloud);
    cropBoxFilter.setMin(minPoint);
    cropBoxFilter.setMax(maxPoint);
    cropBoxFilter.setNegative(false);  // Set to "true" to crop the region and keep the points outside

    // Apply the crop filter to obtain the cropped point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr croppedTargetCloud(new pcl::PointCloud<pcl::PointXYZ>);
    cropBoxFilter.filter(*croppedTargetCloud);

    std::cout << "TEST_1 " << std::to_string(arr_GT_poses[idx][0]) << std::endl;


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

    // Calculating required rigid transform to align the input cloud to the target cloud.
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*output_cloud, transform_GT);

    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
		    << " score: " << ndt.getFitnessScore() << std::endl;

    // Transforming unfiltered, input cloud using found transform.
    pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());

    // Saving transformed input cloud.
    pcl::io::savePCDFileASCII("room_scan2_transformed.pcd", *output_cloud);

    // Initializing point cloud visualizer
    pcl::visualization::PCLVisualizer::Ptr
    viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_final->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ>
    target_color(target_cloud, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud");

    // Coloring and visualizing transformed input cloud (green).
    pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ>
    output_color(output_cloud, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");

    // Starting visualizer
    viewer_final->addCoordinateSystem(1.0, "global");
    viewer_final->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer_final->wasStopped())
    {
        viewer_final->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }

    return (0);
}


