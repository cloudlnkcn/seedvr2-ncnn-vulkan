#include "memory_policy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks=0;
void require(bool result,const char *message) {++checks;if(!result)throw std::runtime_error(message);}
template<class F> void rejects(F action) {
    bool rejected=false;
    try {action();}catch(const std::exception &){rejected=true;}
    require(rejected,"Expected rejection");
}
}
int main() {
    using namespace seedvr2;
    using namespace seedvr2::memory;
    try {
        const MemoryOptions automatic{WeightPlacement::automatic,20};
        require(!choose(automatic,100,DeviceBudget{200,80,0}).host,"Exact boundary must fit");
        require(choose(automatic,100,DeviceBudget{200,81,0}).host,"Usage was not subtracted");
        require(choose(automatic,100,DeviceBudget{200,201,0}).available_bytes==0,"Usage overflow");
        require(choose(automatic,100,DeviceBudget{200,201,0}).host,"Overbudget must use RAM");
        const auto missing=choose(automatic,100,std::nullopt);
        require(missing.host&&!missing.available_bytes,"Unknown budget must not fabricate free memory");
        require(std::string(missing.reason)=="budget_unavailable","Unknown reason lost");
        require(choose({},0,DeviceBudget{0,0,0}).host,"Invalid zero driver budget must be unavailable");
        require(!choose({},150,DeviceBudget{200,0,0}).host,"Default reserve boundary");
        require(choose({},151,DeviceBudget{200,0,0}).host,"Default quarter-heap reserve");
        require(choose({WeightPlacement::automatic,UINT64_MAX},1,DeviceBudget{UINT64_MAX,0,0}).host,"Reserve sum overflow");
        require(choose({WeightPlacement::host,0},1,std::nullopt).host,"Manual host");
        require(!choose({WeightPlacement::device,0},UINT64_MAX,std::nullopt).host,"Manual device");
        require(weight_estimate(12)==24,"Payload estimator");
        require(weight_estimate(UINT64_MAX/2)==UINT64_MAX-1,"Largest valid estimate");
        rejects([]{weight_estimate(UINT64_MAX/2+1);});
        rejects([]{choose({static_cast<WeightPlacement>(100),0},1,std::nullopt);});
        rejects([]{validate({WeightPlacement::host,0},false);});
        rejects([]{validate({WeightPlacement::device,0},false);});
        rejects([]{validate({WeightPlacement::automatic,1},false);});
        rejects([]{validate({WeightPlacement::host,1},true);});
        validate({},false);
        std::cout<<"PASS "<<checks<<" independent budget and precondition checks\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
