#include <cascade/config.h>
// #include <cascade/service_server_api.hpp>
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>


/**
 * This is an prototype for the scheduler dll
 */
namespace derecho{
namespace cascade{




// Helper function
int nthOccurrence(const std::string& str, const std::string& findMe, int nth)
{
    size_t  pos = 0;
    int     cnt = 0;
    while( cnt != nth )
    {
        pos+=1;
        pos = str.find(findMe, pos);
        if ( pos == std::string::npos )
            return -1;
        cnt++;
    }
    return pos;
}


template <typename... CascadeTypes>
std::tuple<std::string, uint32_t, uint32_t> select_shard(CascadeContext<CascadeTypes...>& ctxt, std::string object_pool_id, std::string key){
    uint32_t p_subgroup_index = 0, p_shard_index = 0;
    std::string p_subgroup_type;
    std::tuple<std::string, uint32_t, uint32_t> picked_loc;
    ObjectPoolMetadata obj_pool_meta = ctxt.get_service_client_ref().find_object_pool(object_pool_id);
    if(obj_pool_meta.is_valid()){
        p_subgroup_type = obj_pool_meta.subgroup_type;
        p_subgroup_index = obj_pool_meta.subgroup_index;
        p_subgroup_type = obj_pool_meta.subgroup_type;
        dbg_default_info("[Scheduled] based on object pool");
        uint32_t total_num_shards;
        if(p_subgroup_type=="VCSS"){
            total_num_shards = ctxt.get_service_client_ref().template get_number_of_shards<VolatileCascadeStoreWithStringKey>(p_subgroup_index); 
        }else if (p_subgroup_type=="PCSS"){
            total_num_shards = ctxt.get_service_client_ref().template get_number_of_shards<PersistentCascadeStoreWithStringKey>(p_subgroup_index); 
        }else{
            return std::make_tuple(p_subgroup_type, p_subgroup_index, p_shard_index);
        }
        unsigned int h_key = hash_string_key(key);
        switch(obj_pool_meta.sharding_policy) {
        // only pick shard 0
        case 0:
            p_shard_index = 0;
            break;
        // only pick last shard
        case 1:
            p_shard_index = total_num_shards - 1;
            break;
        // use hashing scheme
        case 2:
            p_shard_index = h_key % total_num_shards; // use time as random source.
            break;
        default:
            throw new derecho::derecho_exception("Unknown member selection policy:" \
                + std::to_string(static_cast<unsigned int>(obj_pool_meta.sharding_policy)) );
            break;
        }
    }
    return std::make_tuple(p_subgroup_type, p_subgroup_index,p_shard_index);;
}


} // namespace cascade
} // namespace derecho

