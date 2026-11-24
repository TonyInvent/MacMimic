#pragma once
#include "inference_cpu_layout.hpp"
#include "policy_gamepad.hpp"
#include <linux/joystick.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstdlib>

namespace sprite_policy {
struct XboxBumpers { int left=-1,right=-1; };
inline XboxBumpers xbox_bumpers(int fd) {
  std::array<unsigned short, KEY_MAX-BTN_MISC+1> codes{};
  unsigned char count=0;
  if(ioctl(fd,JSIOCGBUTTONS,&count)<0 || ioctl(fd,JSIOCGBTNMAP,codes.data())<0)
    throw std::runtime_error("Cannot read Xbox joystick button map");
  XboxBumpers buttons;
  for(int i=0;i<count;++i) {
    if(codes[i]==BTN_TL) buttons.left=i;
    if(codes[i]==BTN_TR) buttons.right=i;
  }
  if(buttons.left<0 || buttons.right<0) throw std::runtime_error("Xbox LB/RB buttons unavailable");
  return buttons;
}
inline std::uint64_t gamepad_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
// All device IO stays off the policy inference thread. No event-age timeout:
// Linux joystick events are change-driven even when a stick is held steadily.
class LinuxGamepad {
  std::atomic<bool> stop_{false};std::mutex mutex_;GamepadSnapshot latest_{};
  std::thread worker_;
  void publish(GamepadSnapshot s){std::lock_guard<std::mutex> lock(mutex_);latest_=s;}
  void run(const std::string& path,bool lateral_enabled,bool kernel_bumpers) {
    GamepadPermission permission(lateral_enabled);
    XboxBumpers buttons;
#ifdef SPRITE_GAMEPAD_STOP_COMBO
    const char* l=std::getenv("SPRITE_GAMEPAD_LB"),*r=std::getenv("SPRITE_GAMEPAD_RB");
    if(!l || !r)throw std::runtime_error("verified LB/RB mapping required");
    const int lb=std::stoi(l),rb=std::stoi(r);
    if(lb<0 || rb<0 || lb>=32 || rb>=32 || lb==rb)throw std::runtime_error("invalid LB/RB mapping");
    buttons={lb,rb};
#endif
    GamepadDisableCombo combo;
    auto disconnected=[&](){combo.disconnect();GamepadSnapshot s;s.disable_requested=combo.sample(gamepad_now());publish(s);};
    while(!stop_) {
      const int fd=open(path.c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);
      if(fd<0){disconnected();std::this_thread::sleep_for(std::chrono::milliseconds(100));continue;}
      unsigned char axes=0;
      bool good=ioctl(fd,JSIOCGAXES,&axes)==0 && axes>=6;
      if(kernel_bumpers) {
        try { buttons=xbox_bumpers(fd); }
        catch(...) { ::close(fd); GamepadSnapshot fault;fault.button_map_fault=true;publish(fault);return; }
      }
      permission.disconnect();
      while(good && !stop_) {
        pollfd p{fd,POLLIN,0};const int rc=poll(&p,1,10);
        if(rc<0 || p.revents&(POLLERR|POLLHUP|POLLNVAL)){good=false;break;}
        for(int n=0;n<256 && rc>0;++n) {
          js_event event{};const auto size=read(fd,&event,sizeof(event));
          if(size<0 && (errno==EAGAIN||errno==EWOULDBLOCK))break;
          if(size!=sizeof(event)){good=false;break;}
          if(event.type&JS_EVENT_AXIS)permission.event(event.number,std::clamp(event.value/32767.,-1.,1.),event.type&JS_EVENT_INIT);
          if((event.type&JS_EVENT_BUTTON) && (event.number==buttons.left || event.number==buttons.right))
            combo.event(event.number==buttons.left,event.value!=0,event.type&JS_EVENT_INIT);
        }
        int version=0;
        if(!good || ioctl(fd,JSIOCGVERSION,&version)<0){good=false;break;}
        auto snapshot=permission.snapshot(gamepad_now());
        snapshot.disable_requested=combo.sample(snapshot.stamp);
        publish(snapshot);
      }
      permission.disconnect();disconnected();close(fd);
    }
  }
public:
  explicit LinuxGamepad(const std::string& path,bool lateral_enabled=false,bool kernel_bumpers=false) {
    if(kernel_bumpers) {
      int fd=open(path.c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);
      if(fd<0) throw std::runtime_error("Cannot open Xbox controller for LB/RB verification");
      try { xbox_bumpers(fd); } catch(...) { ::close(fd);throw; }
      ::close(fd);
    }
    worker_=std::thread([this,path,lateral_enabled,kernel_bumpers]{
      try { run(path,lateral_enabled,kernel_bumpers); }
      catch(...) { GamepadSnapshot fault;fault.button_map_fault=kernel_bumpers;publish(fault); }
    });
  }
  ~LinuxGamepad(){stop_=true;if(worker_.joinable())worker_.join();}
    void pin_worker(int cpu) { sprite_schedule::pin_thread(worker_.native_handle(),cpu); }
  GamepadSnapshot snapshot(){std::lock_guard<std::mutex> lock(mutex_);return latest_;}
};
}
