#include <iostream>
#include <thread>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}

int
main(int argc, char** argv)
{
  // initialize PointClouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
  // populate our PointCloud with points
  pcl::io::loadPCDFile ("lane1.pcd", *cloud);
  pcl::io::loadPCDFile ("1554884725.127955000.pcd", *cloud2);
  int size1=(*cloud).points.size();
  int size2=(*cloud2).points.size();
  (*cloud2).points.resize(size1+size2);
  for(int i=0;i<size1;i++){
   (*cloud2).points[size2+i]=(*cloud).points[i];
  }
  std::vector<int> inliers;
  Eigen::VectorXf coef;
  // created RandomSampleConsensus object and compute the appropriated model
  
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
  if(pcl::console::find_argument (argc, argv, "-f") >= 0)
  {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
    ransac.getModelCoefficients(coef);

  }
  

  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);

  // creates the visualization object and adds either our original cloud or all of the inliers
  // depending on the command line arguments specified.
  pcl::visualization::PCLVisualizer::Ptr viewer;
  if (pcl::console::find_argument (argc, argv, "-f") >= 0)
    {viewer = simpleVis(final);
    cout<<"a="<<coef(0)<<" b="<<coef(1)<<" c="<<coef(2)<<" d="<<coef(3)<<endl;
    pcl::io::savePCDFileASCII ("plane.pcd", *final);
}
  else
    viewer = simpleVis(cloud2);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
  return 0;
 }
