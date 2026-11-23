#include "adaptive_window/gaussian_journal.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sys/stat.h>
#include <cmath>
#include <iomanip>
#include <locale>

namespace fast_livo { namespace adaptive_window {
namespace {
void pose(std::ostream& s, const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
  s << '[' << p.x() << ',' << p.y() << ',' << p.z() << ',' << q.x() << ','
    << q.y() << ',' << q.z() << ',' << q.w() << ']';
}
}
GaussianJournal::GaussianJournal(const std::string& directory): dir_(directory) {
  if (::mkdir(dir_.c_str(), 0755) != 0) {
    ROS_ERROR_STREAM("Gaussian journal needs a NEW directory under an existing parent: " << dir_); return;
  }
  out_.open(dir_ + "/events.jsonl");
  if (!good()) { ROS_ERROR_STREAM("Cannot write Gaussian journal: " << dir_); return; }
  out_.imbue(std::locale::classic());
  out_ << std::setprecision(17) << "{\"kind\":\"header\",\"version\":1}";
  finishLine();
}
void GaussianJournal::finishLine() {
  out_ << '\n' << std::flush;
  if (!good()) ROS_ERROR_STREAM("Gaussian journal write failed: " << dir_);
}
GaussianJournal::~GaussianJournal() { if (good()) { out_ << "{\"kind\":\"end\"}"; finishLine(); } }

void GaussianJournal::scan(double t, const StatesGroup& proposal, const LidarExtrinsic& ex,
                            const std::vector<PointToPlane>& planes) {
  if (!good() || !std::isfinite(t)) return;
  out_ << "{\"kind\":\"scan\",\"t\":" << t << ",\"rows\":[";
  bool first=true;
  // Keep packet size bounded. Use the deskewed return in IMU coordinates;
  // proposal rotation expresses the matched surface normal in I only.
  const size_t step = std::max<size_t>(1,(planes.size()+2047)/2048);
  for (size_t i=0;i<planes.size();i+=step) {
    const auto& pl=planes[i];
    Eigen::Vector3d p=ex.q_i_l*pl.point_b_+ex.p_i_l;
    Eigen::Vector3d n=proposal.rot_end.transpose()*pl.normal_;
    Eigen::Vector3d n_l=ex.q_i_l.conjugate()*n;
    if (!p.allFinite() || !n.allFinite() || n.norm()<1e-6 || !pl.body_cov_.allFinite()) continue;
    n.normalize(); n_l.normalize();
    const double var=n_l.dot(pl.body_cov_*n_l);
    if (!std::isfinite(var) || var<0) continue;
    if (!first) out_ << ',';
    first=false;
    out_ << '[' << p.x()<<','<<p.y()<<','<<p.z()<<','<<n.x()<<','<<n.y()<<','<<n.z()
         <<','<<std::max(1e-6,var)<<']';
  }
  out_ << "]}"; finishLine();
}

void GaussianJournal::image(double t, const VIOManagerPtr& v) {
  if (!good() || !std::isfinite(t) || !v || !v->cam || v->img_rgb.empty()
      || !(v->fx>0 && v->fy>0)) return;
  const auto size=v->img_rgb.size();
  try {
    // The upstream VIO image may still be distorted. Resample from its actual
    // projection model to an explicitly declared, half-resolution pinhole.
    const cv::Size dst(std::max(1,size.width/2),std::max(1,size.height/2));
    if (map_x_.size()!=dst) {
      map_x_.create(dst,CV_32F); map_y_.create(dst,CV_32F);
      for (int y=0;y<dst.height;++y) for (int x=0;x<dst.width;++x) {
        Eigen::Vector3d ray((2*x+0.5-v->cx)/v->fx,(2*y+0.5-v->cy)/v->fy,1);
        const auto px=v->cam->world2cam(ray);
        map_x_.at<float>(y,x)=px.x(); map_y_.at<float>(y,x)=px.y();
      }
    }
    cv::Mat rect,valid;
    cv::remap(v->img_rgb,rect,map_x_,map_y_,cv::INTER_LINEAR);
    const std::string key=std::to_string(++serial_), filename="rgb_"+key+".png";
    if (serial_==1 || last_size_!=dst) {
      valid=(map_x_>=0)&(map_x_<size.width-1)&(map_y_>=0)&(map_y_<size.height-1);
      // Keep masks immutable across resolution changes.
      mask_id_=key; last_size_=dst;
      if (!cv::imwrite(dir_+"/valid_"+mask_id_+".png",valid)) throw std::runtime_error("mask write");
    }
    if (!cv::imwrite(dir_+"/"+filename,rect)) throw std::runtime_error("image write");
    out_ << "{\"kind\":\"image\",\"t\":"<<t<<",\"file\":\""<<filename
         <<"\",\"mask\":\"valid_"<<mask_id_<<".png\",\"K\":["<<v->fx/2<<','<<v->fy/2<<','
         <<(v->cx-0.5)/2<<','<<(v->cy-0.5)/2<<"]}"; finishLine();
  } catch (const std::exception& ex) {
    ROS_ERROR_STREAM("Gaussian image recording failed: " << ex.what());
    out_.setstate(std::ios_base::badbit);
  }
}
void GaussianJournal::solution(const Estimator& e, bool accepted) {
  if (!good()) return;
  if (epoch_ != e.diagnostics().reset_count) {
    epoch_=e.diagnostics().reset_count;
    out_ << "{\"kind\":\"reset\",\"epoch\":"<<epoch_<<'}'; finishLine();
  }
  if (!accepted) return;
  const auto ex=e.cameraExtrinsicEstimate();
  out_ << "{\"kind\":\"solve\",\"T_CI\":"; pose(out_,ex.p_c_i,ex.q_c_i);
  out_ << ",\"poses\":[";
  bool first=true;
  for (const auto& f:e.mappingSnapshot()) {
    if (!first) out_<<',';
    first=false;
    out_<<"{\"id\":"<<f.id<<",\"t\":"<<f.state.timestamp<<",\"T\":";
    pose(out_,f.state.p_w_i,f.state.q_w_i);
    out_<<",\"active\":"<<(e.hasFrame(f.id)?"true":"false")<<",\"P\":";
    if (!f.covariance_valid) out_<<"null";
    else {
      out_<<'['; for(int i=0;i<36;++i) { if(i) out_<<','; out_<<f.covariance(i/6,i%6); } out_<<']';
    }
    out_<<'}';
  }
  out_<<"]}"; finishLine();
}
}}
