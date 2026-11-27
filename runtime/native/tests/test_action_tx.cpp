#include "../src/action_tx_trace.hpp"
#include <cassert>
#include <iostream>
int main() {
  using namespace sprite_policy;
  // An accepted action commits in the current transport slot.
  // No skipped sequence is admissible.
  for (std::size_t arrival=0; arrival<400; ++arrival) {
    auto slot=arrival;
    while (!coherent_commit_due(slot,2,1)) ++slot;
    assert(slot==arrival);
    assert(!coherent_commit_due(slot,2,2));
  }
  for (auto pending : {0u,3u}) {
    bool rejected=false;
    try { (void)coherent_commit_due(4,pending,1); }
    catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
  }
  ActionTxTrace trace;trace.prepare();
  trace.context={2,10,20,true,false};
  trace.sent(7,30,31);trace.sent(8,32,33);
  // Hardware and software completions may arrive separately and out of order.
  trace.stamp(1,0,500);trace.stamp(0,400,0);trace.stamp(1,401,0);trace.stamp(0,0,499);
  assert(trace.events[0].hardware_raw_ns==499 && trace.events[1].software_realtime_ns==401);
  assert(trace.events[1].context.sequence==2 && !trace.events[1].context.powered);
  bool rejected=false;
  try {trace.stamp(2,1,0);} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
  const std::array<std::uint8_t,8> payload{1,2,3,4,5,6,7,8};
  ActionTxTrace echo;
  echo.prepare();
  echo.context={2,10,20,true,false};
  echo.sent(7,30,31,0,payload.data());
  auto wrong=payload;
  wrong[7]=0;
  rejected=false;
  try {echo.confirm_echo(7,wrong.data(),40);} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
  echo.confirm_echo(7,payload.data(),42);
  assert(echo.events[0].echo_received_ns==42 && echo.events[0].hardware_raw_ns==0);
  echo.sent(8,50,51,0,payload.data());
  rejected=false;
  try {echo.confirm_echo(8,payload.data(),20'000'010);} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
  // Exceed the old limit while retaining every packet identity without reallocation.
  const auto* allocation = trace.events.data();
  for (int i=2; i<650000; ++i) trace.sent(7, 30, 31);
  assert(trace.events.data()==allocation && trace.events.size()==650000);
  trace.stamp(649999, 600, 700);
  assert(trace.events.back().hardware_raw_ns==700);
  std::cout << "coherent sequence and TX identity tests passed\n";
}
