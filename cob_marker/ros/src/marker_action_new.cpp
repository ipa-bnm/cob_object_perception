/****************************************************************
 *
 * Copyright (c) 2010
 *
 * Fraunhofer Institute for Manufacturing Engineering
 * and Automation (IPA)
 *
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * Project name: care-o-bot
 * ROS stack name: cob_object_perception
 * ROS package name: cob_fiducials
 *
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * Author: Jan Fischer, email:jan.fischer@ipa.fhg.de
 * Supervised by: Jan Fischer, email:jan.fischer@ipa.fhg.de
 *
 * Date of creation: March 2013
 * ToDo:
 *
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * Redistribution and use in source and binary rforms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Fraunhofer Institute for Manufacturing
 *       Engineering and Automation (IPA) nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License LGPL as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License LGPL for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License LGPL along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

//##################
//#### includes ####

// standard includes
//--

// ROS includes
#include <ros/ros.h>
//#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <actionlib/server/simple_action_server.h>

#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl/ros/conversions.h>
#include <pcl/common/pca.h>

// ROS message includes
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// external includes
#include <cob_object_detection_msgs/DetectObjects.h>
#include <cob_object_detection_msgs/DetectObjectsAction.h>
#include <cob_object_detection_msgs/DetectObjectsActionGoal.h>
#include <cob_object_detection_msgs/DetectObjectsActionResult.h>

#include <cob_marker/general_marker.h>

#include <boost/thread/mutex.hpp>
#include <boost/timer.hpp>
#include <boost/functional/hash.hpp>

//#include "opencv/highgui.h"

using namespace message_filters;

namespace cob_marker
{

typedef sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::PointCloud2, sensor_msgs::CameraInfo> ColorDepthSyncPolicy;


/// @class CobFiducialsNode
/// This node gathers images from a 'color camera'
/// to recognize fiducials
class CobMarkerNode //: public nodelet::Nodelet
{
    struct MarkerDetectorParams
    {
        std::string detector_algo;
        bool try_harder;
        double marker_size;
        double focal_length;
        double timeout;
        int max_markers;
        std::string frame_id;
    };

private:
    ros::NodeHandle node_handle_;

    boost::shared_ptr<image_transport::ImageTransport> image_transport_0_;
    boost::shared_ptr<image_transport::ImageTransport> image_transport_1_;

    // Subscriptions
    image_transport::SubscriberFilter color_camera_image_sub_;	///< color camera image topic
    message_filters::Subscriber<sensor_msgs::PointCloud2> depth_camera_image_sub_;	///< camera information service
    message_filters::Subscriber<sensor_msgs::CameraInfo> color_camera_info_sub_;

    boost::shared_ptr<message_filters::Synchronizer<ColorDepthSyncPolicy > > color_image_sub_sync_; ///< Synchronizer
    tf::TransformListener transform_listener_; ///< tf transforms

    int sub_counter_; /// Number of subscribers to topic
    unsigned int endless_counter_; ///< A counter to show that node is still receiving images
    //bool synchronizer_received_; ///< Set to true, when synchronizer fires

    // Action definitions
    typedef actionlib::SimpleActionServer<cob_object_detection_msgs::DetectObjectsAction> ActionServer;
    ActionServer* detect_marker_action_;
    // Service definitions
    ros::ServiceServer detect_marker_service_; ///< Service server to request fidcuial detection
    // Publisher definitions
    ros::Publisher detect_marker_pub_;
    ros::Publisher marker_marker_array_publisher_;
    image_transport::Publisher img2D_pub_; ///< Publishes 2D image data to show detection results

    ros::Time received_timestamp_;
    std::string received_frame_id_;

    cv::Mat camera_matrix_;
    bool camera_matrix_initialized_;

    bool publish_tf_;
    tf::TransformBroadcaster tf_broadcaster_; ///< Broadcast transforms of detected fiducials
    bool publish_2d_image_;
    bool publish_marker_array_; ///< Publish coordinate systems of detected fiducials as marker for rviz
    unsigned int prev_marker_array_size_; ///< Size of previously published marker array
    visualization_msgs::MarkerArray marker_array_msg_;

    static std::string color_image_encoding_;
    bool publisher_enabled_; ///< Set if marker publisher should be enabled
    bool service_enabled_; ///< Set if marker service server should be enabled
    bool action_enabled_; ///< Set if marker action server should be enabled

    MarkerDetectorParams detectorParams_;

    boost::mutex mutexQ_;
    boost::condition_variable condQ_;

    boost::shared_ptr<GeneralMarker> m_marker_detector;
    sensor_msgs::PointCloud2ConstPtr buffered_point_cloud_;
    sensor_msgs::ImageConstPtr buffered_image_;

public:
    /// Constructor.
    CobMarkerNode(ros::NodeHandle& nh)
        : sub_counter_(0),
          endless_counter_(0),
          camera_matrix_initialized_(false)
    {
        /// Void
        node_handle_ = nh;
        onInit();
    }

    /// Destructor.
    ~CobMarkerNode()
    {
        marker_marker_array_publisher_.shutdown();
    }

    void onInit()
    {
        /// Create a handle for this node, initialize node
        //	  node_handle_ = getMTNodeHandle();
        image_transport_0_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(node_handle_));
        image_transport_1_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(node_handle_));

        /// Initialize camera node
        if (!init()) return;
    }

    /// Initialize sensor fusion node.
    /// Setup publisher of point cloud and corresponding color data,
    /// setup camera toolboxes and colored point cloud toolbox
    /// @return <code>true</code> on success, <code>false</code> otherwise
    bool init()
    {
        if (loadParameters() == false) return false;

        ros::SubscriberStatusCallback imgConnect    = boost::bind(&CobMarkerNode::connectCallback, this);
        ros::SubscriberStatusCallback imgDisconnect = boost::bind(&CobMarkerNode::disconnectCallback, this);

        // Synchronize inputs of incoming image data.
        // Topic subscriptions happen on demand in the connection callback.
        ROS_INFO("[cob_marker] Setting up image data subscribers");
        if (publisher_enabled_)
        {
            detect_marker_pub_ = node_handle_.advertise<cob_object_detection_msgs::DetectionArray>("detect_marker", 1, imgConnect, imgDisconnect);
        }
        if (service_enabled_)
        {
            detect_marker_service_ = node_handle_.advertiseService("get_marker", &CobMarkerNode::detectMarkerServiceCallback, this);
        }
        if (action_enabled_)
        {
            detect_marker_action_ = new ActionServer(node_handle_, "marker_detection", boost::bind(&CobMarkerNode::detectMarkerActionCallback, this, _1), false);
        }

        // Publisher for visualization/debugging
        marker_marker_array_publisher_ = node_handle_.advertise<visualization_msgs::MarkerArray>( "marker_marker_array", 0 );
        img2D_pub_= image_transport_1_->advertise("image", 1);

        //synchronizer_received_ = false;
        prev_marker_array_size_ = 0;

        ROS_INFO("[cob_marker] Setting up marker detector library");
        m_marker_detector = boost::shared_ptr<GeneralMarker>(setupMarkerDetector());

        ROS_INFO("[cob_marker] Initializing [OK]");
        ROS_INFO("[cob_marker] Up and running");
        return true;
    }

    GeneralMarker* setupMarkerDetector()
    {
        GeneralMarker* detector = NULL;
        if(detectorParams_.detector_algo.compare("zxing") == 0)
        {
            Marker_Zxing* zxing = new Marker_Zxing();
            zxing->setTryHarder(detectorParams_.try_harder);
            detector = zxing;
        }
        else if(detectorParams_.detector_algo.compare("dmtx") == 0)
        {
            Marker_DMTX* dmtx = new Marker_DMTX();
            dmtx->setTimeout((int)(detectorParams_.timeout*1000));
            dmtx->setMaxDetectionCount(detectorParams_.max_markers);
            detector = dmtx;
        }
        ROS_ASSERT(detector != NULL);
        return detector;
    }


    /// Subscribe to camera topics if not already done.
    void connectCallback()
    {
        ROS_INFO("[cob_marker] Subscribing to camera topics");

        color_camera_image_sub_.subscribe(*image_transport_0_, "image_color", 1);
        depth_camera_image_sub_.subscribe(node_handle_, "point_cloud", 1);
        color_camera_info_sub_.subscribe(node_handle_, "camera_info", 1);

        color_image_sub_sync_ = boost::shared_ptr<message_filters::Synchronizer<ColorDepthSyncPolicy> >(new message_filters::Synchronizer<ColorDepthSyncPolicy>(ColorDepthSyncPolicy(2)));
        color_image_sub_sync_->connectInput(color_camera_image_sub_, depth_camera_image_sub_, color_camera_info_sub_);
        color_image_sub_sync_->registerCallback(boost::bind(&CobMarkerNode::cameraSyncCallback, this, _1, _2, _3));

        sub_counter_++;
        ROS_INFO("[cob_marker] %i subscribers on camera topics [OK]", sub_counter_);
    }

    /// Unsubscribe from camera topics if possible.
    void disconnectCallback()
    {
        if (sub_counter_ > 0)
        {
            ROS_INFO("[cob_marker] Unsubscribing from camera topics");

            color_camera_image_sub_.unsubscribe();
            depth_camera_image_sub_.unsubscribe();
            color_camera_info_sub_.unsubscribe();

            sub_counter_--;
            ROS_INFO("[cob_marker] %i subscribers on camera topics [OK]", sub_counter_);
        }
    }

    // void depthCallback(const sensor_msgs::PointCloud2ConstPtr& msg_depth)
    // {
    //     ROS_INFO("depthCallback");
    // }

    /// Callback is executed, when shared mode is selected
    /// Left and right is expressed when facing the back of the camera in horizontal orientation.
    void cameraSyncCallback(const sensor_msgs::ImageConstPtr& color_camera_data,
                            const sensor_msgs::PointCloud2ConstPtr& msg_depth,
                            const sensor_msgs::CameraInfoConstPtr& color_camera_info)
    {
        ROS_INFO("[cob_marker] color image callback");
        {
            boost::mutex::scoped_lock lock( mutexQ_ );


            if (camera_matrix_initialized_ == false)
            {
                camera_matrix_ = cv::Mat::zeros(3,3,CV_64FC1);
                camera_matrix_.at<double>(0,0) = color_camera_info->K[0];
                camera_matrix_.at<double>(0,2) = color_camera_info->K[2];
                camera_matrix_.at<double>(1,1) = color_camera_info->K[4];
                camera_matrix_.at<double>(1,2) = color_camera_info->K[5];
                camera_matrix_.at<double>(2,2) = 1;

                ROS_INFO("[cob_marker] Initializing fiducial detector with camera matrix");
                camera_matrix_initialized_ = true;
            }

            // Receive
            received_timestamp_ = color_camera_data->header.stamp;
            received_frame_id_ = color_camera_data->header.frame_id;
            buffered_point_cloud_ = msg_depth;
            buffered_image_ = color_camera_data;


            if (publisher_enabled_ == true)
            {
                cob_object_detection_msgs::DetectionArray detection_array;
                detectMarkers(detection_array, buffered_point_cloud_, buffered_image_);

                // Publish
                detect_marker_pub_.publish(detection_array);

                // cv_bridge::CvImage cv_ptr;
                // cv_ptr.image = color_mat_8U3_;
                // cv_ptr.encoding = CobFiducialsNode::color_image_encoding_;
                // img2D_pub_.publish(cv_ptr.toImageMsg());
            }

            //synchronizer_received_ = true;

            // Notify waiting thread
        }
        condQ_.notify_one();
    }



    bool detectMarkerServiceCallback(cob_object_detection_msgs::DetectObjects::Request &req,
                                        cob_object_detection_msgs::DetectObjects::Response &res)
    {
        ROS_INFO("[cob_marker] Service Callback");
        // Connect to image topics
        bool result = false;
        //synchronizer_received_ = false;
        connectCallback();
        const double action_timeout = 5.;

        // Wait for data
        {
            boost::mutex::scoped_lock lock( mutexQ_);
            boost::system_time const timeout=boost::get_system_time()+ boost::posix_time::milliseconds(5000);

            ROS_INFO("[cob_marker] Waiting for image data");
            if (condQ_.timed_wait(lock, timeout))
                ROS_INFO("[cob_marker] Waiting for image data [OK]");
            else
            {
                ROS_WARN("[cob_marker] Could not receive image data from ApproximateTime synchronizer");
                return false;
            }

            // Wait for data (at least 5 seconds)
            //int nSecPassed = 0;
            //float nSecIncrement = 0.5;
            //while (!synchronizer_received_ && nSecPassed < 10)
            //{
            //	ros::Duration(nSecIncrement).sleep();
            //	nSecPassed += nSecIncrement;
            //	ROS_INFO("[fiducials] Waiting");
            //}

            //if (!synchronizer_received_)
            //{
            //	ROS_WARN("[fiducials] Could not receive image data");
            //	return false;
            //}
            double time_start = ros::Time::now().toSec();

            while(ros::Time::now().toSec()-time_start<action_timeout || result != true)
            {
                result = detectMarkers(res.object_list, buffered_point_cloud_, buffered_image_);
            }
        }
        disconnectCallback();

        return result;
    }

    void detectMarkerActionCallback(const cob_object_detection_msgs::DetectObjectsGoalConstPtr &goal)
    {
        ROS_INFO("[cob_marker] Action Callback");
        bool result = false;
        connectCallback();
        cob_object_detection_msgs::DetectObjectsResult res;

        const double action_timeout = 5.;

        // Wait for data
        {
            boost::mutex::scoped_lock lock( mutexQ_);
            boost::system_time const timeout=boost::get_system_time()+ boost::posix_time::milliseconds(5000);

            ROS_INFO("[cob_marker] Waiting for image data");
            if (condQ_.timed_wait(lock, timeout))
                ROS_INFO("[cob_marker] Waiting for image data [OK]");
            else
            {
                ROS_WARN("[cob_marker] Could not receive image data from ApproximateTime synchronizer");
                result = false;
            }

            if (detect_marker_action_->isPreemptRequested() || !ros::ok())
            {
                detect_marker_action_->setPreempted();
                return;
            }
            double time_start = ros::Time::now().toSec();

            while(ros::Time::now().toSec()-time_start<action_timeout || result != true)
            {
                result = detectMarkers(res.object_list, buffered_point_cloud_, buffered_image_);
            }
            if(result)
                detect_marker_action_->setSucceeded(res);
            else
                detect_marker_action_->setAborted(res);
        }

        disconnectCallback();
    }

    bool compPCA(pcl::PCA<pcl::PointCloud<pcl::PointXYZ>::PointType> &pca, const pcl::PointCloud<pcl::PointXYZ> &pc, const float w, const Eigen::Vector2i &o, const Eigen::Vector2f &d) {
        pcl::PointCloud<pcl::PointXYZ> tmp_pc;
        for(int x=0; x<w; x++) {
          Eigen::Vector2i p = o + (d*x).cast<int>();
          if(pcl_isfinite(pc(p(0),p(1)).getVector3fMap().sum()))
            tmp_pc.push_back( pc(p(0),p(1)) );
        }
        if(tmp_pc.size()<3) {
          ROS_WARN("no valid points");
          return false;
        }
        tmp_pc.width=1;
        tmp_pc.height=tmp_pc.size();
        pca.setInputCloud(tmp_pc.makeShared());
    return true;
  }

    bool detectMarkers(cob_object_detection_msgs::DetectionArray& detection_array,
        sensor_msgs::PointCloud2ConstPtr point_cloud,
        sensor_msgs::ImageConstPtr image)
    {
        std::stringstream ss;
        std::vector<GeneralMarker::SMarker> res;
        unsigned int marker_array_size = 0;
        unsigned int pose_array_size = 0;

        double time_before_find = ros::Time::now().toSec();
        bool found = m_marker_detector->findPattern(*image, res);
        ROS_INFO("[cob_marker] findPattern: runtime %f s ; %d pattern found", (ros::Time::now().toSec() - time_before_find), (int)res.size());
        if(found)
        {
            pose_array_size = res.size();
            pcl::PointCloud<pcl::PointXYZ> pc;
            pcl::fromROSMsg(*point_cloud, pc);

            for(size_t i=0; i<res.size(); i++)
            {
              //get 6DOF pose
              if(res[i].pts_.size()<3)
              {
                ROS_WARN("need 3 points");
                continue;
              }

              /*
               *   1---3
               *   |   |
               *   0---2
               */
              Eigen::Vector2f d1 = (res[i].pts_[1]-res[i].pts_[0]).cast<float>();
              Eigen::Vector2f d2 = (res[i].pts_[2]-res[i].pts_[0]).cast<float>();

              ROS_DEBUG("p1: %d %d", res[i].pts_[0](0), res[i].pts_[0](1)); 
              ROS_DEBUG("p2: %d %d", res[i].pts_[1](0), res[i].pts_[1](1)); 
              ROS_DEBUG("p3: %d %d", res[i].pts_[2](0), res[i].pts_[2](1)); 
              ROS_DEBUG("p4: %d %d", res[i].pts_[3](0), res[i].pts_[3](1)); 
              
              ROS_DEBUG("d1: %f %f", d1(0), d1(1)); 
              ROS_DEBUG("d2: %f %f", d2(0), d2(1)); 

              int w1=std::max(std::abs(d1(0)),std::abs(d1(1)));
              int w2=std::max(std::abs(d2(0)),std::abs(d2(1)));
              d1/=w1;
              d2/=w2;

              pcl::PCA<pcl::PointCloud<pcl::PointXYZ>::PointType> pca1, pca2;
              if(!compPCA(pca1, pc, w1, res[i].pts_[0],d1))
                continue;
              if(!compPCA(pca2, pc, w2, res[i].pts_[0],d2))
                continue;

              int i1=0;
              if(pca1.getEigenValues()[1]>pca1.getEigenValues()[i1]) i1=1;
              if(pca1.getEigenValues()[2]>pca1.getEigenValues()[i1]) i1=2;
              int i2=0;
              if(pca2.getEigenValues()[1]>pca2.getEigenValues()[i2]) i2=1;
              if(pca2.getEigenValues()[2]>pca2.getEigenValues()[i2]) i2=2;

              if(pca1.getEigenVectors().col(i1).sum()<0)
                pca1.getEigenVectors().col(i1)*=-1;
              if(pca2.getEigenVectors().col(i2).sum()<0)
                pca2.getEigenVectors().col(i2)*=-1;

              Eigen::Vector3f m = (pca1.getMean()+pca2.getMean()).head<3>()/2;
              Eigen::Matrix3f M, M2;
              M.col(0) = pca2.getEigenVectors().col(i2);
              M.col(1) = M.col(0).cross((Eigen::Vector3f)pca1.getEigenVectors().col(i1));
              M.col(1).normalize();
              M.col(2) = M.col(0).cross(M.col(1));

              Eigen::Quaternionf q(M);
              M2 = M;
              M2.col(1)=M.col(2);
              M2.col(2)=M.col(1);
              Eigen::Quaternionf q2(M);

              //TODO: please change to ROS_DEBUG
              ss.clear();
              ss.str("");
              ss<<"E\n"<<pca1.getEigenVectors()<<"\n";
              ss<<"E\n"<<pca2.getEigenVectors()<<"\n";
              ss<<"E\n"<<pca1.getEigenValues()<<"\n";
              ss<<"E\n"<<pca2.getEigenValues()<<"\n";
              ROS_DEBUG("%s",ss.str().c_str());
              
              ss.clear();
              ss.str("");
              ss<<"M\n"<<M2<<"\n";
              ss<<"d\n"<<M.col(0).dot(M.col(1))<<"\n";
              ss<<"d\n"<<M.col(0).dot(M.col(2))<<"\n";
              ss<<"d\n"<<M.col(1).dot(M.col(2))<<"\n";
              ROS_DEBUG("%s",ss.str().c_str());
              //std::cout<<"m\n"<<m<<"\n";

              cob_object_detection_msgs::Detection det;
              det.header = point_cloud->header;
              det.label = res[i].code_.substr(0,3);
              det.detector = m_marker_detector->getName();
              det.pose.header = point_cloud->header;
              det.pose.pose.position.x = m(0);
              det.pose.pose.position.y = m(1);
              det.pose.pose.position.z = m(2);
              det.pose.pose.orientation.w = q2.w();
              det.pose.pose.orientation.x = q2.x();
              det.pose.pose.orientation.y = q2.y();
              det.pose.pose.orientation.z = q2.z();
              detection_array.detections.push_back(det);

              ROS_INFO("[cob_marker] Detected Tag: '%s' at x,y,z,rw,rx,ry,rz ( %f, %f, %f, %f, %f, %f, %f )", det.label.c_str(),
                        det.pose.pose.position.x,det.pose.pose.position.y,det.pose.pose.position.z,
                        det.pose.pose.orientation.w, det.pose.pose.orientation.x, det.pose.pose.orientation.y, det.pose.pose.orientation.z);
            }    
            //Publish 2d imagebuffered_point_cloud_
            if (publish_2d_image_)
            {
                // Receive
                cv_bridge::CvImageConstPtr cv_ptr;
                try
                {
                  cv_ptr = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
                }
                catch (cv_bridge::Exception& e)
                {
                  ROS_ERROR("cv_bridge exception: %s", e.what());
                  return false;
                }

                cv::Mat color_image = cv_ptr->image;

                for (unsigned int i=0; i<pose_array_size; i++)
                {
                    std::vector<double> pose(7, 0.0);
                    pose[0] = detection_array.detections[i].pose.pose.position.x;
                    pose[1] = detection_array.detections[i].pose.pose.position.y;
                    pose[2] = detection_array.detections[i].pose.pose.position.z;
                    pose[3] = detection_array.detections[i].pose.pose.orientation.w;
                    pose[4] = detection_array.detections[i].pose.pose.orientation.x;
                    pose[5] = detection_array.detections[i].pose.pose.orientation.y;
                    pose[6] = detection_array.detections[i].pose.pose.orientation.z;
                    cv::Mat rot_3x3;
                    cv::Mat trans_3x1;
                    cv::Mat frame_4x4 = Vec7ToFrame(pose);

                    rot_3x3 = frame_4x4(cv::Rect(0, 0, 3, 3));
                    trans_3x1 = frame_4x4(cv::Rect(3, 0, 1, 3));

                    RenderPose(color_image, rot_3x3, trans_3x1);

                    cv_bridge::CvImage cv_ptr;
                    cv_ptr.image = color_image;
                    cv_ptr.encoding = CobMarkerNode::color_image_encoding_;
                    img2D_pub_.publish(cv_ptr.toImageMsg());
                }
            }
            
            // Publish tf
            if (publish_tf_)
            {
                for (unsigned int i=0; i<pose_array_size; i++)
                {
                    // Broadcast transform of fiducial
                    tf::Transform transform;
                    std::stringstream tf_name;
                    tf_name << "cob_marker_tag" <<"_" << res[i].code_;
                    transform.setOrigin(tf::Vector3(detection_array.detections[i].pose.pose.position.x,
                        detection_array.detections[i].pose.pose.position.y,
                        detection_array.detections[i].pose.pose.position.z));
                    transform.setRotation(tf::Quaternion(detection_array.detections[i].pose.pose.orientation.w,
                        detection_array.detections[i].pose.pose.orientation.x,
                        detection_array.detections[i].pose.pose.orientation.y,
                        detection_array.detections[i].pose.pose.orientation.z));
                    tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), received_frame_id_, tf_name.str()));
                }
            }

            // Publish marker array
            if (publish_marker_array_)
            {
                // 3 arrows for each coordinate system of each detected fiducial
                marker_array_size = 3*pose_array_size;
                if (marker_array_size >= prev_marker_array_size_)
                {
                    marker_array_msg_.markers.resize(marker_array_size);
                }

                boost::hash<std::string> string_hash;
                // publish a coordinate system from arrow markers for each object
                for (unsigned int i=0; i<pose_array_size; i++)
                {
                    for (unsigned int j=0; j<3; j++)
                    {
                        unsigned int idx = 3*i+j;
                        marker_array_msg_.markers[idx].header.frame_id = received_frame_id_;// "/" + frame_id;//"tf_name.str()";
                        marker_array_msg_.markers[idx].header.stamp = received_timestamp_;
                        marker_array_msg_.markers[idx].ns = "cob_marker";
                        marker_array_msg_.markers[idx].id =  string_hash(res[i].code_);
                        marker_array_msg_.markers[idx].type = visualization_msgs::Marker::ARROW;
                        marker_array_msg_.markers[idx].action = visualization_msgs::Marker::ADD;
                        marker_array_msg_.markers[idx].color.a = 0.85;
                        marker_array_msg_.markers[idx].color.r = 0;
                        marker_array_msg_.markers[idx].color.g = 0;
                        marker_array_msg_.markers[idx].color.b = 0;

                        marker_array_msg_.markers[idx].points.resize(2);
                        marker_array_msg_.markers[idx].points[0].x = 0.0;
                        marker_array_msg_.markers[idx].points[0].y = 0.0;
                        marker_array_msg_.markers[idx].points[0].z = 0.0;
                        marker_array_msg_.markers[idx].points[1].x = 0.0;
                        marker_array_msg_.markers[idx].points[1].y = 0.0;
                        marker_array_msg_.markers[idx].points[1].z = 0.0;

                        if (j==0)
                        {
                            marker_array_msg_.markers[idx].points[1].x = 0.2;
                            marker_array_msg_.markers[idx].color.r = 255;
                        }
                        else if (j==1)
                        {
                            marker_array_msg_.markers[idx].points[1].y = 0.2;
                            marker_array_msg_.markers[idx].color.g = 255;
                        }
                        else if (j==2)
                        {
                            marker_array_msg_.markers[idx].points[1].z = 0.2;
                            marker_array_msg_.markers[idx].color.b = 255;
                        }

                        marker_array_msg_.markers[idx].pose = detection_array.detections[i].pose.pose;
       

                        ros::Duration one_hour = ros::Duration(10); // 1 second
                        marker_array_msg_.markers[idx].lifetime = one_hour;
                        marker_array_msg_.markers[idx].scale.x = 0.01; // shaft diameter
                        marker_array_msg_.markers[idx].scale.y = 0.015; // head diameter
                        marker_array_msg_.markers[idx].scale.z = 0; // head length 0=default
                    }

                    if (prev_marker_array_size_ > marker_array_size)
                    {
                        for (unsigned int i = marker_array_size; i < prev_marker_array_size_; ++i)
                        {
                            marker_array_msg_.markers[i].action = visualization_msgs::Marker::DELETE;
                        }
                    }
                    prev_marker_array_size_ = marker_array_size;

                    marker_marker_array_publisher_.publish(marker_array_msg_);
                }
            }
        } // End: publish markers

        if (res.empty())
            return false;
        return true;
    }

    unsigned long RenderPose(cv::Mat& image, cv::Mat& rot_3x3_CfromO, cv::Mat& trans_3x1_CfromO)
    {
        cv::Mat object_center(3, 1, CV_64FC1);
        double* p_object_center = object_center.ptr<double>(0);
        p_object_center[0] = 0;
        p_object_center[1] = 0;
        p_object_center[2] = 0;

        cv::Mat rot_inv = rot_3x3_CfromO.inv();

        // Compute coordinate axis for visualization
        cv::Mat pt_axis(4, 3, CV_64FC1);
        double* p_pt_axis = pt_axis.ptr<double>(0);
        p_pt_axis[0] = 0 + p_object_center[0];
        p_pt_axis[1] = 0 + p_object_center[1];
        p_pt_axis[2] = 0 + p_object_center[2];
        p_pt_axis = pt_axis.ptr<double>(1);
        p_pt_axis[0] = 0.1 + p_object_center[0];
        p_pt_axis[1] = 0 + p_object_center[1];
        p_pt_axis[2] = 0 + p_object_center[2];
        p_pt_axis = pt_axis.ptr<double>(2);
        p_pt_axis[0] = 0 + p_object_center[0];
        p_pt_axis[1] = 0.1 + p_object_center[1];
        p_pt_axis[2] = 0 + p_object_center[2];
        p_pt_axis = pt_axis.ptr<double>(3);
        p_pt_axis[0] = 0 + p_object_center[0];
        p_pt_axis[1] = 0 + p_object_center[1];
        p_pt_axis[2] = 0.1 + p_object_center[2];

        // Transform data points
        std::vector<cv::Point> vec_2d(4, cv::Point());
        for (int i=0; i<4; i++)
        {
            cv::Mat vec_3d = pt_axis.row(i).clone();
            vec_3d = vec_3d.t();
            vec_3d = rot_3x3_CfromO*vec_3d;
            vec_3d += trans_3x1_CfromO;
            double* p_vec_3d = vec_3d.ptr<double>(0);

            ReprojectXYZ(p_vec_3d[0], p_vec_3d[1], p_vec_3d[2],
                         vec_2d[i].x , vec_2d[i].y);
        }

        // Render results
        int line_width = 1;
        cv::line(image, vec_2d[0], vec_2d[1], cv::Scalar(0, 0, 255), line_width);
        cv::line(image, vec_2d[0], vec_2d[2], cv::Scalar(0, 255, 0), line_width);
        cv::line(image, vec_2d[0], vec_2d[3], cv::Scalar(255, 0, 0), line_width);

        return 1;
    }

    unsigned long ReprojectXYZ(double x, double y, double z, int& u, int& v)
    {
        cv::Mat XYZ(3, 1, CV_64FC1);
        cv::Mat UVW(3, 1, CV_64FC1);

        double* d_ptr = 0;
        double du = 0;
        double dv = 0;
        double dw = 0;

        x *= 1000;
        y *= 1000;
        z *= 1000;

        d_ptr = XYZ.ptr<double>(0);
        d_ptr[0] = x;
        d_ptr[1] = y;
        d_ptr[2] = z;

        UVW = camera_matrix_ * XYZ;

        d_ptr = UVW.ptr<double>(0);
        du = d_ptr[0];
        dv = d_ptr[1];
        dw = d_ptr[2];

        u = cvRound(du/dw);
        v = cvRound(dv/dw);

        return 1;
    }

// Function copied from cob_vision_ipa_utils/MathUtils.h to avoid dependency
    inline float SIGN(float x)
    {
        return (x >= 0.0f) ? +1.0f : -1.0f;
    }
    std::vector<double> FrameToVec7(const cv::Mat frame)
    {
        // [0]-[2]: translation xyz
        // [3]-[6]: quaternion wxyz
        std::vector<double> pose(7, 0.0);

        double r11 = frame.at<double>(0,0);
        double r12 = frame.at<double>(0,1);
        double r13 = frame.at<double>(0,2);
        double r21 = frame.at<double>(1,0);
        double r22 = frame.at<double>(1,1);
        double r23 = frame.at<double>(1,2);
        double r31 = frame.at<double>(2,0);
        double r32 = frame.at<double>(2,1);
        double r33 = frame.at<double>(2,2);

        double qw = ( r11 + r22 + r33 + 1.0) / 4.0;
        double qx = ( r11 - r22 - r33 + 1.0) / 4.0;
        double qy = (-r11 + r22 - r33 + 1.0) / 4.0;
        double qz = (-r11 - r22 + r33 + 1.0) / 4.0;
        if(qw < 0.0f) qw = 0.0;
        if(qx < 0.0f) qx = 0.0;
        if(qy < 0.0f) qy = 0.0;
        if(qz < 0.0f) qz = 0.0;
        qw = std::sqrt(qw);
        qx = std::sqrt(qx);
        qy = std::sqrt(qy);
        qz = std::sqrt(qz);
        if(qw >= qx && qw >= qy && qw >= qz)
        {
            qw *= +1.0;
            qx *= SIGN(r32 - r23);
            qy *= SIGN(r13 - r31);
            qz *= SIGN(r21 - r12);
        }
        else if(qx >= qw && qx >= qy && qx >= qz)
        {
            qw *= SIGN(r32 - r23);
            qx *= +1.0;
            qy *= SIGN(r21 + r12);
            qz *= SIGN(r13 + r31);
        }
        else if(qy >= qw && qy >= qx && qy >= qz)
        {
            qw *= SIGN(r13 - r31);
            qx *= SIGN(r21 + r12);
            qy *= +1.0;
            qz *= SIGN(r32 + r23);
        }
        else if(qz >= qw && qz >= qx && qz >= qy)
        {
            qw *= SIGN(r21 - r12);
            qx *= SIGN(r31 + r13);
            qy *= SIGN(r32 + r23);
            qz *= +1.0;
        }
        else
        {
            printf("coding error\n");
        }
        double r = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= r;
        qx /= r;
        qy /= r;
        qz /= r;

        pose[3] = qw;
        pose[4] = qx;
        pose[5] = qy;
        pose[6] = qz;

        // Translation
        pose[0] = frame.at<double>(0,3);
        pose[1] = frame.at<double>(1,3);
        pose[2] = frame.at<double>(2,3);
        return pose;
    }
    cv::Mat Vec7ToFrame(const std::vector<double>& pose)
    {
        // Assumption for ipa_Utils::Vec7d pose
        // [0]-[2]: translation xyz
        // [3]-[6]: quaternion wxyz

        cv::Mat trans_mat;
        trans_mat.create( 4, 4, CV_64FC1);

        // normalize
        const double n = 1.0f/sqrt(pose[3]*pose[3]+pose[4]*pose[4]+pose[5]*pose[5]+pose[6]*pose[6]);
        double qx = n * pose[4];
        double qy = n * pose[5];
        double qz = n * pose[6];
        double qw = n * pose[3];

        trans_mat.at<double>(0, 0) = 1 - 2*(qy*qy) - 2*(qz*qz);
        trans_mat.at<double>(0, 1) = 2 * (qx*qy-qw*qz);
        trans_mat.at<double>(0, 2) = 2 * (qx*qz+qw*qy);
        trans_mat.at<double>(0, 3) = pose[0];
        trans_mat.at<double>(1, 0) = 2 * (qx*qy+qw*qz);
        trans_mat.at<double>(1, 1) = 1 - 2*(qx*qx) - 2*(qz*qz);
        trans_mat.at<double>(1, 2) = 2 * (qy*qz-qw*qx);
        trans_mat.at<double>(1, 3) = pose[1];
        trans_mat.at<double>(2, 0) = 2 * (qx*qz-qw*qy);
        trans_mat.at<double>(2, 1) = 2 * (qy*qz+qw*qx);
        trans_mat.at<double>(2, 2) = 1 - 2*(qx*qx) - 2*(qy*qy);
        trans_mat.at<double>(2, 3) = pose[2];
        trans_mat.at<double>(3, 0) = 0;
        trans_mat.at<double>(3, 1) = 0;
        trans_mat.at<double>(3, 2) = 0;
        trans_mat.at<double>(3, 3) = 1;
        return trans_mat;
    }

    bool loadParameters()
    {
        std::string tmp_string;

        /// Parameters are set within the launch file
        node_handle_.param<bool>("publisher_enabled", publisher_enabled_, false);
        ROS_INFO("[cob_marker] ROS publisher enabled: %u", publisher_enabled_);

        node_handle_.param<bool>("service_enabled", service_enabled_, false);
        ROS_INFO("[cob_marker] ROS service enabled: %u", service_enabled_);

        node_handle_.param<bool>("action_enabled", action_enabled_, true);
        ROS_INFO("[cob_marker] ROS action enabled: %u", action_enabled_);

        node_handle_.param<std::string>("algorithm",detectorParams_.detector_algo,"dmtx");
        ROS_INFO("[cob_marker] using %s algorithm", detectorParams_.detector_algo.c_str());

        node_handle_.param<double>("dmtx_timeout",detectorParams_.timeout, 0.5);
        ROS_INFO("[cob_marker] dmtx_timeout: %f", detectorParams_.timeout);

        node_handle_.param<int>("dmtx_max_markers",detectorParams_.max_markers, 2);
        ROS_INFO("[cob_marker] dmtx_max_markers: %i", detectorParams_.max_markers);

        node_handle_.param<std::string>("frame_id",detectorParams_.frame_id,"/head_cam3d_link");
        ROS_INFO("[cob_marker] frame_id: %s", detectorParams_.frame_id.c_str());

        node_handle_.param<bool>("try_harder",detectorParams_.try_harder, true);
        ROS_INFO("[cob_marker] try_harder: %u", detectorParams_.try_harder);
        
        node_handle_.param<bool>("publish_marker_array", publish_marker_array_, false);
        if (publish_marker_array_)
            ROS_INFO("[cob_marker] publish_marker_array: true");
        else
            ROS_INFO("[cob_marker] publish_marker_array: false");

        node_handle_.param<bool>("publish_tf", publish_tf_, false);
        if (publish_tf_)
            ROS_INFO("[cob_marker] publish_tf: true");
        else
            ROS_INFO("[cob_marker] publish_tf: false");

        node_handle_.param<bool>("publish_2d_image", publish_2d_image_, false);
        if (publish_2d_image_)
            ROS_INFO("[cob_marker] publish_2d_image: true");
        else
            ROS_INFO("[cob_marker] publish_2d_image: false");



        return true;
    }
};
std::string CobMarkerNode::color_image_encoding_ = "bgr8";
}; // END namepsace
//1: The namespace in which the Triangle plugin will live. Typically, we use the name of the package
// that contains the library that Triangle is a part of. In this case, that's pluginlib_tutorials
// which is the name of the package we created in step one of this tutorial.
//2: The name we wish to give to the plugin.... we'll call ours regular_triangle.
//3: The fully-qualified type of the plugin class, in this case, polygon_plugins::Triangle.
//4: The fully-qualified type of the base class, in this case, polygon_base::RegularPolygon
// plugin/nodelet namespace, plugin/nodelet name, qualified class name, qualified nodelete class name
//PLUGINLIB_DECLARE_CLASS(cob_sensor_fusion, cob_sensor_fusion_nodelet, ipa_SensorFusion::CobSensorFusionNode, nodelet::Nodelet);

//#######################
//#### main programm ####
int main(int argc, char** argv)
{
    /// initialize ROS, specify name of node
    ros::init(argc, argv, "cob_marker");

    /// Create a handle for this node, initialize node
    ros::NodeHandle nh("~");

    /// Create camera node class instance
    cob_marker::CobMarkerNode marker_node(nh);

    ros::MultiThreadedSpinner spinner(2); // Use 4 threads
    spinner.spin();
    //	ros::spin();

    return 0;
}
