#include <cascade/config.h>
#include <cascade/service_server_api.hpp>
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

template <typename... CascadeTypes>
std::pair<uint32_t, uint32_t> schedule_to_shard(ServiceClient<CascadeTypes...>& service_client_ref, std::string object_pool_id, std::string key){
    uint32_t p_subgroup_index = 0, p_shard_index = 0;
    std::pair<uint32_t, uint32_t> picked_loc;
    ObjectPoolMetadata obj_pool_meta = service_client_ref.find_object_pool(object_pool_id);
    if(obj_pool_meta.is_valid()){
        p_subgroup_index = obj_pool_meta.subgroup_index;
        uint32_t total_num_shards = service_client_ref.get_number_of_shards<SubgroupType>(subgroup_index);
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
    picked_loc.first = p_subgroup_index;
    picked_loc.second = p_shard_index;
    return picked_loc;
}

} // namespace cascade
} // namespace derecho

