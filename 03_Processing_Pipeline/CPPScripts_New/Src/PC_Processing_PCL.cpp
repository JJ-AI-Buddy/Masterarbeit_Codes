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
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/gasd.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/features/gfpfh.h>
#include <pcl/features/normal_based_signature.h>
#include <pcl/features/boost.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/grsd.h>
#include <pcl/features/moment_invariants.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/vfh.h>
#include <pcl/features/fpfh.h>

#include <windows.h>

using namespace std::chrono_literals;
using namespace std;
//namespace fs = std::filesystem;



int
main()
{
    string path_pc = "C:\\Users\\Johanna\\OneDrive - bwedu\\Masterarbeit_OSU\\Data\\Dataset_01_10122023\\Route_1_Scan_08.pcd";

    string dataset = "Route_1";
    string timestamp = "XXXX";
    string input_cloud = "Scan_08";
    string feature_descriptor;
    string number_input_points;
    string feature_descriptor_size;

    


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_pc, *cloud) == -1) {
        PCL_ERROR("Couldn't read file containing the point cloud\n");
        return (-1);
    }

    // Verify the loaded point cloud
    if (!cloud->empty()) {
        std::cout << "Loaded " << cloud->size() << " points from the PCD file." << std::endl;
    }
    else {
        std::cerr << "Failed to load the point cloud." << std::endl;
        return (-1);
    }


    ///////////////////////////////
    // Hand-crafted features
    //////////////////////////////

    // Keypoints / Feature detectors

    // // Harris 3D (Corners)

    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
    harris.setInputCloud(cloud);
    // Set other parameters as needed
    harris.setNonMaxSupression(true);
    harris.setRadius(2.5);

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_harris3d(new pcl::PointCloud<pcl::PointXYZI>);
    harris.compute(*keypoints_harris3d);

    std::cout << "Computed " << keypoints_harris3d->size() << " Harris3D keypoints from the preprocessed point cloud." << std::endl;


    // // ISS 3D 

    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZRGBA> iss;
    iss.setInputCloud(cloud);
    // Set other parameters as needed
    iss.setSalientRadius(2.8);
    iss.setNonMaxRadius(2.5);
    iss.setThreshold21(0.95);
    iss.setThreshold32(0.95);
    iss.setMinNeighbors(5);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints_iss3d (new pcl::PointCloud<pcl::PointXYZRGBA>);
    iss.compute(*keypoints_iss3d);

    std::cout << "Computed " << keypoints_iss3d->size() << " ISS-3D keypoints from the preprocessed point cloud." << std::endl;

    // Colorize keypoints
    uint8_t r = 0;
    uint8_t g = 255;
    uint8_t b = 0;

    for (auto& point : keypoints_iss3d->points) {
        point.r = r;
        point.g = g;
        point.b = b;
    }

    // // SIFT - Scale Invariant Feature Transform keypoints - not working !!!!

    //pcl::SIFTKeypoint<pcl::PointXYZI, pcl::PointXYZRGBA> sift;
    //sift.setInputCloud(cloud);
    //Set other parameters as needed
    //sift.setMinimumContrast(0.2);
    //sift.setScales();

    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints_sift3d(new pcl::PointCloud<pcl::PointXYZRGBA>);
    //sift.compute(*keypoints_sift3d);

    //std::cout << "Computed " << keypoints_sift3d->size() << " SIFT-3D keypoints from the preprocessed point cloud." << std::endl;

    // Colorize keypoints
    //r = 0;
    //g = 0;
    //b = 255;

    //for (auto& point : keypoints_sift3d->points) {
    //    point.r = r;
    //    point.g = g;
    //    point.b = b;
    //}


    // Compute surface normals

    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_processed (new pcl::PointCloud<pcl::PointXYZ>);


    // Convert from pcl::PointXYZI to pcl::PointXYZ
    keypoints_processed->resize(keypoints_harris3d->size());

    for (size_t i = 0; i < keypoints_harris3d->size(); ++i) {
        keypoints_processed->points[i].x = keypoints_harris3d->points[i].x;
        keypoints_processed->points[i].y = keypoints_harris3d->points[i].y;
        keypoints_processed->points[i].z = keypoints_harris3d->points[i].z;
    }

    number_input_points = std::to_string(keypoints_processed->size());

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(keypoints_processed);
    ne.setRadiusSearch(1.5);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // Prepare csv-file data

    std::vector<std::string> header_csv_base = {
        "Feature descriptor",
        "Dataset",
        "Timestamp",
        "Input Cloud",
        "Number Input points",
        "Size feature descriptor"   
    };

    std::string filename_csv;


    // Feature descriptors

    // CVFHEstimation estimates the Clustered Viewpoint Feature Histogram (CVFH) descriptor
    // https://pcl.readthedocs.io/projects/tutorials/en/master/vfh_estimation.html


    filename_csv = "CVFH_Feature.csv";

    // Extend the csv-header

    std::vector<std::string> header_csv;
    header_csv = header_csv_base;


    for (int j = 0; j < 308; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    std::ofstream file(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "CVFH feature csv-file has been created as " << filename_csv << std::endl;

    pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(keypoints_processed);
    cvfh.setClusterTolerance(0.02); 
    cvfh.setInputNormals(normals);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhs_feature(new pcl::PointCloud<pcl::VFHSignature308>);
    cvfh.compute(*cvfhs_feature);

    std::cout << "Computed " << cvfhs_feature->size() << " CVFH feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    std::cout << "CVFH feature vector with 308 entries: " << cvfhs_feature << "." << std::endl;


    feature_descriptor = "CVFH-Histogram";
    feature_descriptor_size = "1x308";

    // Initialize string vector as first part of data to write to csv
    std::vector<std::string> data_csv_string = {
        feature_descriptor,
        dataset,
        timestamp,
        input_cloud,
        number_input_points,
        feature_descriptor_size
    };

    //Open the csv-file in append-mode
    std::ofstream file_add(filename_csv, std::ios_base::app);

    if (!file_add) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Iterate through the point cloud and output the VFH signatures histogram
    for (size_t i = 0; i < cvfhs_feature->size(); ++i) {
        const pcl::VFHSignature308& cvfh_signature = cvfhs_feature->points[i];

        // Loop through the data and write it to the file in CSV format
        for (size_t i = 0; i < data_csv_string.size(); ++i) {
            file_add << data_csv_string[i];
            if (i < data_csv_string.size() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }

        file_add << ";";

        // Access the elements of the VFH signature and output them
        for (int j = 0; j < 308; ++j) {
            std::cout << "Element " << j << ": " << cvfh_signature.histogram[j] << std::endl;

            file_add << cvfh_signature.histogram[j];

            if (j < cvfh_signature.descriptorSize() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << "\n";
    }

    //Close file
    file_add.close();

    // GASD - Globally Aligned Spatial Distribution 

    filename_csv = "GASD_Feature.csv";

    // Extend the csv-header

    header_csv = header_csv_base;

    for (int j = 0; j < 512; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    //file(filename_csv);
    //file << filename_csv;
    file.open(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "GASD feature csv-file has been prepared as" << filename_csv << std::endl;

    pcl::GASDEstimation<pcl::PointXYZ, pcl::GASDSignature512> gasd;
    gasd.setInputCloud(keypoints_processed);
    //gasd.setRadiusSearch(0.3); 

    pcl::PointCloud<pcl::GASDSignature512>::Ptr gasd_descriptors(new pcl::PointCloud<pcl::GASDSignature512>);
    gasd.compute(*gasd_descriptors);

    std::cout << "Computed " << gasd_descriptors->size() << " GASD feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    std::cout << "GASD feature vector with 512 entries: " << gasd_descriptors << "." << std::endl;


    feature_descriptor = "GASD-Histogram";
    feature_descriptor_size = std::to_string(gasd_descriptors->size()) + "x512";

    // Initialize string vector as first part of data to write to csv
    data_csv_string = {
        feature_descriptor,
        dataset,
        timestamp,
        input_cloud,
        number_input_points,
        feature_descriptor_size
    };

    //Open the csv-file in append-mode
    //file_add(filename_csv, std::ios_base::app);
    //file_add << filename_csv;
    file_add.open(filename_csv, std::ios_base::app);

    if (!file_add) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Iterate through the point cloud and output the GASD descriptor
    for (size_t i = 0; i < gasd_descriptors->size(); ++i) {
        const pcl::GASDSignature512& gasd_signature = gasd_descriptors->points[i];

        // Loop through the data and write it to the file in CSV format
        for (size_t i = 0; i < data_csv_string.size(); ++i) {
            file_add << data_csv_string[i];
            if (i < data_csv_string.size() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << ";";

        // Access the elements of the GASD signature and output them
        for (int j = 0; j < 512; ++j) {
            std::cout << "Element " << j << ": " << gasd_signature.histogram[j] << std::endl;

            file_add << gasd_signature.histogram[j];

            if (j < gasd_signature.descriptorSize() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << "\n";
    }

    //Close file
    file_add.close();

    // GFPFH - Global Fast Point Feature Histogram - not working; Linker error

    //pcl::GFPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::GFPFHSignature16> gfpfh;
    //gfpfh.setInputCloud(keypoints_processed);
    ////gfpfh.setInputNormals(normals);
    ////gfpfh.setOctreeLeafSize(0.05);
    ////gfpfh.setRadiusSearch(0.1); 

    //pcl::PointCloud<pcl::GFPFHSignature16>::Ptr gfpfh_descriptors(new pcl::PointCloud<pcl::GFPFHSignature16>);
    //gfpfh.compute(*gfpfh_descriptors);
   

    //std::cout << "Computed " << gfpfh_descriptors->size() << " GFPFH feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    //std::cout << "GFPFH feature vector with 16 entries: " << gfpfh_descriptors << "." << std::endl;


    //// Iterate through the point cloud and output the GFPFH descriptor
    //for (size_t i = 0; i < gfpfh_descriptors->size(); ++i) {
    //    const pcl::GFPFHSignature16& gfpfh_signature = gfpfh_descriptors->points[i];


    //    // Access the elements of the VFH signature and output them
    //    for (int j = 0; j < 16; ++j) {
    //        std::cout << "Element " << j << ": " << gfpfh_signature.histogram[j] << std::endl;
    //    }
    //}


    // GRSD - Global Radius-based Surface 

    filename_csv = "GRSD_Feature.csv";

    // Extend the csv-header

    header_csv = header_csv_base;

    for (int j = 0; j < 21; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    //file(filename_csv);
    //file << filename_csv;
    file.open(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "GRSD feature csv-file has been prepared as" << filename_csv << std::endl;

    pcl::GRSDEstimation<pcl::PointXYZ, pcl::Normal, pcl::GRSDSignature21> grsd;
    grsd.setInputCloud(keypoints_processed);
    grsd.setRadiusSearch(0.3);
    grsd.setInputNormals(normals);

    pcl::PointCloud<pcl::GRSDSignature21>::Ptr grsd_descriptors(new pcl::PointCloud<pcl::GRSDSignature21>);
    grsd.compute(*grsd_descriptors);

    std::cout << "Computed " << grsd_descriptors->size() << " GRSD feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    std::cout << "GRSD feature vector with 21 entries: " << grsd_descriptors << "." << std::endl;

    feature_descriptor = "GRSD-Histogram";
    feature_descriptor_size = std::to_string(gasd_descriptors->size()) + "x21";

    // Initialize string vector as first part of data to write to csv
    data_csv_string = {
        feature_descriptor,
        dataset,
        timestamp,
        input_cloud,
        number_input_points,
        feature_descriptor_size
    };

    //Open the csv-file in append-mode
    //file_add(filename_csv, std::ios_base::app);
    //file_add << filename_csv;
    file_add.open(filename_csv, std::ios_base::app);

    if (!file_add) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Iterate through the point cloud and output the GRSD descriptor
    for (size_t i = 0; i < grsd_descriptors->size(); ++i) {
        const pcl::GRSDSignature21& grsd_signature = grsd_descriptors->points[i];
        
        // Loop through the data and write it to the file in CSV format
        for (size_t i = 0; i < data_csv_string.size(); ++i) {
            file_add << data_csv_string[i];
            if (i < data_csv_string.size() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << ";";


        // Access the elements of the GRSD signature and output them
        for (int j = 0; j < 21; ++j) {
            std::cout << "Element " << j << ": " << grsd_signature.histogram[j] << std::endl;

            file_add << grsd_signature.histogram[j];

            if (j < grsd_signature.descriptorSize() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << "\n";
    }

    //Close file
    file_add.close();



    // MomentInvariantsEstimation estimates the 3 moment invariants (j1, j2, j3) at each 3D point

    filename_csv = "MomentInvariants.csv";

    // Extend the csv-header

    header_csv = header_csv_base;

    for (int j = 0; j < 3; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    //file(filename_csv);
    //file << filename_csv;
    file.open(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "Moment Invariants feature csv-file has been prepared as" << filename_csv << std::endl;

    pcl::MomentInvariantsEstimation<pcl::PointXYZ, pcl::MomentInvariants> mominv;
    mominv.setInputCloud(keypoints_processed);
    mominv.setRadiusSearch(0.1);

    pcl::PointCloud<pcl::MomentInvariants>::Ptr mominv_descriptors(new pcl::PointCloud<pcl::MomentInvariants>);
    mominv.compute(*mominv_descriptors);

    std::cout << "Computed " << mominv_descriptors->size() << " vector(s) of moment invariants from Harris 3D keypoints of the original point cloud." << std::endl;

    feature_descriptor = "Moment_Invariants";
    feature_descriptor_size = std::to_string(gasd_descriptors->size()) + "x3";

    // Initialize string vector as first part of data to write to csv
    data_csv_string = {
        feature_descriptor,
        dataset,
        timestamp,
        input_cloud,
        number_input_points,
        feature_descriptor_size
    };

    //Open the csv-file in append-mode
    //file_add(filename_csv, std::ios_base::app);
    //file_add << filename_csv;
    file_add.open(filename_csv, std::ios_base::app);

    if (!file_add) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }


    // Iterate through the point cloud and output the Moment Invariants descriptor
    for (size_t i = 0; i < mominv_descriptors->size(); ++i) {
        const pcl::MomentInvariants& mominv_signature = mominv_descriptors->points[i];

        // Loop through the data and write it to the file in CSV format
        for (size_t i = 0; i < data_csv_string.size(); ++i) {
            file_add << data_csv_string[i];
            if (i < data_csv_string.size() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << ";";


        std::cout << "Element j1 : " << mominv_signature.j1 << std::endl;
        std::cout << "Element j2 : " << mominv_signature.j2 << std::endl;
        std::cout << "Element j3 : " << mominv_signature.j3 << std::endl;

        file_add << mominv_signature.j1;
        file_add << ";";
        file_add << mominv_signature.j2;
        file_add << ";";
        file_add << mominv_signature.j3;

        file_add << "\n";
    }


    // SHOT - Signature of Histograms of OrienTations (SHOT) descriptor - not properly working!!!!

    pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::ReferenceFrame> lrf_estimation;
    lrf_estimation.setInputCloud(keypoints_processed);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference_frames(new pcl::PointCloud<pcl::ReferenceFrame>);
    lrf_estimation.compute(*reference_frames);

    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setInputCloud(keypoints_processed);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(5);
    shot.setInputReferenceFrames(reference_frames);

    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors(new pcl::PointCloud<pcl::SHOT352>);
    shot.compute(*shot_descriptors);

    std::cout << "Computed " << shot_descriptors->size() << " vector(s) of SHOT descriptors from Harris 3D keypoints of the original point cloud." << std::endl;
    

    // SpinImage - Spin image is a histogram of point locations summed along the bins of the image

    pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>> spin_image;
    spin_image.setInputCloud(keypoints_processed);
    spin_image.setInputNormals(normals);
    spin_image.setRadiusSearch(0.1);


    pcl::PointCloud<pcl::Histogram<153>>::Ptr spin_images(new pcl::PointCloud<pcl::Histogram<153>>);
    spin_image.compute(*spin_images);

    std::cout << "Computed " << spin_images->size() << " vector(s) of Spin Images descriptors from Harris 3D keypoints of the original point cloud." << std::endl;



    // Iterate through the point cloud and output the SpinImage descriptor
    //for (size_t i = 0; i < spin_images->size(); ++i) {
    //    const pcl::Histogram<153>& spin_images_signature = spin_images->points[i];


    //    // Access the elements of the SpinImage signature and output them
    //    for (int j = 0; j < 153; ++j) {
    //        std::cout << "Element " << j << ": " << spin_images_signature.histogram[j] << std::endl;
    //    }
    //}


    // VFH - Viewpoint Feature Histogram

    filename_csv = "VFH_Feature.csv";

    // Extend the csv-header

    header_csv = header_csv_base;

    for (int j = 0; j < 308; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    //file(filename_csv);
    //file << filename_csv;
    file.open(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "VFH feature csv-file has been prepared as" << filename_csv << std::endl;


    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(keypoints_processed);
    vfh.setInputNormals(normals);
    // Optionally, set parameters using vfh.set...

    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(new pcl::PointCloud<pcl::VFHSignature308>);
    vfh.compute(*vfhs);

    std::cout << "Computed " << vfhs->size() << " VFH feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    std::cout << "VFH feature vector with 308 entries: " << vfhs << "." << std::endl;

    feature_descriptor = "VFH-Histogram";
    feature_descriptor_size = std::to_string(gasd_descriptors->size()) + "x308";

    // Initialize string vector as first part of data to write to csv
    data_csv_string = {
        feature_descriptor,
        dataset,
        timestamp,
        input_cloud,
        number_input_points,
        feature_descriptor_size
    };

    //Open the csv-file in append-mode
    //file_add(filename_csv, std::ios_base::app);
    //file_add << filename_csv;
    file_add.open(filename_csv, std::ios_base::app);

    if (!file_add) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Iterate through the point cloud and output the VFH descriptor
    for (size_t i = 0; i < vfhs->size(); ++i) {
        const pcl::VFHSignature308& vfhs_signature = vfhs->points[i];

        // Loop through the data and write it to the file in CSV format
        for (size_t i = 0; i < data_csv_string.size(); ++i) {
            file_add << data_csv_string[i];
            if (i < data_csv_string.size() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << ";";

        // Access the elements of the VFH signature and output them
        for (int j = 0; j < 308; ++j) {
            std::cout << "Element " << j << ": " << vfhs_signature.histogram[j] << std::endl;

            file_add << vfhs_signature.histogram[j];

            if (j < vfhs_signature.descriptorSize() - 1) {
                file_add << ";"; // Add a comma as a separator between values
            }
        }
        file_add << "\n";
    }

    file_add.close();


    // FPFH - Fast Point Feature Histogram

    filename_csv = "FPFH_Feature.csv";

    // Extend the csv-header

    header_csv = header_csv_base;

    for (int j = 0; j < 33; ++j) {
        std::string str_add;

        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << j;
        str_add = oss.str();
        header_csv.push_back(str_add);
    }

    // Open the file for writing
    //file(filename_csv);
    //file << filename_csv;
    file.open(filename_csv);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the csv.-file for writing." << std::endl;
        return 1; // Exit with an error code
    }

    // Loop through the data and write it to the file in CSV format
    for (size_t i = 0; i < header_csv.size(); ++i) {
        file << header_csv[i];
        if (i < header_csv.size() - 1) {
            file << ";"; // Add a comma as a separator between values
        }
    }
    file << "\n";

    //Close file
    file.close();
    std::cout << "FPFH feature csv-file has been prepared as" << filename_csv << std::endl;


    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypoints_processed);
    fpfh.setInputNormals(normals);
    fpfh.setRadiusSearch(0.1);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.compute(*fpfhs);

    std::cout << "Computed " << fpfhs->size() << " FPFH feature vector(s) from Harris 3D keypoints of the original point cloud." << std::endl;
    std::cout << "FPFH feature vector with 33 entries: " << fpfhs << "." << std::endl;

    

    // Iterate through the point cloud and output the FPFH descriptor
    //for (size_t i = 0; i < fpfhs->size(); ++i) {
    //    const pcl::FPFHSignature33& fpfhs_signature = fpfhs->points[i];


    //    // Access the elements of the FPFH signature and output them
    //    for (int j = 0; j < 33; ++j) {
    //        std::cout << "Element " << j << ": " << fpfhs_signature.histogram[j] << std::endl;
    //    }
    //}


    // Initialize the PCL Visualizer
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");

    // Display the point cloud in the viewer
    int v1(0);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.setBackgroundColor(0, 0, 0, v1);
    viewer.addText("Harris 3D Keypoints", 10, 10, "v1 text", v1);
    viewer.addPointCloud<pcl::PointXYZI>(keypoints_harris3d, "sample cloud1", v1);

    int v2(0);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer.setBackgroundColor(0.0, 0.0, 0.0, v2);
    viewer.addText("ISS 3D Keypoints", 10, 10, "v2 text", v2);;
    viewer.addPointCloud<pcl::PointXYZRGBA>(keypoints_iss3d, "sample cloud2");

    /*int v3(0);
    viewer.createViewPort(1.0, 0.5, 1.0, 1.0, v3);
    viewer.setBackgroundColor(0.0, 0.0, 0.0, v3);
    viewer.addText("SIFT 3D Keypoints", 10, 10, "v3 text", v3);;
    viewer.addPointCloud<pcl::PointXYZRGBA>(keypoints_sift3d, "sample cloud3");*/

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud3");

    // Enter the visualization loop (press 'q' to exit)
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }


    return 0;
}


