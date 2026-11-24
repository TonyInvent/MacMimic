#include "../src/policy_gamepad.hpp"
#include "../src/policy_state_ipc.hpp"
#include <cassert>
#include <cstdlib>

int main() {
    sprite_policy::GamepadDisableCombo combo;
    std::uint64_t now=1000000;
    auto hold=[&](int ms) {
        for(int i=0;i<ms;++i) { now+=1000000; combo.sample(now); }
        return combo.sample(now);
    };
    combo.event(true,true,true);
    combo.event(false,true,true);
    assert(!hold(4000));  // Already held when the joystick was opened.
    combo.event(true,false,false);
    combo.event(false,false,false);
    combo.event(true,true,false);
    assert(!hold(4000));  // A single bumper is insufficient.
    combo.event(false,true,false);
    combo.sample(now);
    assert(!hold(2999));
    combo.event(true,false,false);
    combo.event(true,true,false);
    combo.sample(now);
    assert(!hold(2999));
    combo.disconnect();
    assert(!hold(3000));
    combo.event(true,false,true);
    combo.event(false,false,true);
    combo.event(false,true,false);
    combo.event(true,true,false);
    combo.sample(now);
    assert(!hold(2999));
    assert(hold(1));
    combo.event(true,false,false);
    combo.disconnect();
    assert(combo.sample(now));

    char directory[]="/tmp/xbox-disable-XXXXXX";
    assert(mkdtemp(directory));
    const auto path=std::string(directory)+"/controller";
    sockaddr_un address{};
    address.sun_family=AF_UNIX;
    std::strcpy(address.sun_path,path.c_str());
    int server=socket(AF_UNIX,SOCK_SEQPACKET,0);
    assert(server>=0 && bind(server,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
    assert(listen(server,1)==0);
    sprite_policy::PolicyStateChannel channel(path,1,2,20000000);
    int peer=accept(server,nullptr,nullptr);
    assert(peer>=0);
    try { channel.request_operator_disable(); assert(false); }
    catch(const std::runtime_error& e) { assert(e.what()==std::string(sprite_policy::operator_disable_reason)); }
    char packet[1280];
    const auto n=recv(peer,packet,sizeof(packet),0);
    assert(n==8 && sprite_policy::is_operator_disable(packet,n));
    assert(!sprite_policy::is_operator_disable(packet,n-1));
    packet[6]=2;  // Normal impedance target kind.
    assert(!sprite_policy::is_operator_disable(packet,n));
    assert(recv(peer,packet,sizeof(packet),0)==0);
    close(peer);
    close(server);
    unlink(path.c_str());
    rmdir(directory);
}
