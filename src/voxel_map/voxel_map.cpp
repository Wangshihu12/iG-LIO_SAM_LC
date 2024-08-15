#include "voxel_map/voxel_map.h"
namespace lio
{
    // 比较 pair 中的第二个元素
    bool compare(std::pair<Eigen::Vector3d, double> &p1, std::pair<Eigen::Vector3d, double> &p2)
    {
        return p1.second < p2.second;
    }

    // 哈希函数，通过给定三维点输出哈希表中的索引
    size_t HashUtil::operator()(const Eigen::Vector3d &point)
    {
        double loc_xyz[3];
        for (size_t i = 0; i < 3; ++i)
        {
            loc_xyz[i] = point[i] * resolution_inv_;
            if (loc_xyz[i] < 0)
            {
                loc_xyz[i] -= 1.0;
            }
        }
        size_t x = static_cast<size_t>(loc_xyz[0]);
        size_t y = static_cast<size_t>(loc_xyz[1]);
        size_t z = static_cast<size_t>(loc_xyz[2]);

        return ((((z)*hash_p_) % max_n_ + (y)) * hash_p_) % max_n_ + (x);
    }

    // 设置最大最小值
    void Grid::setMinMax(size_t min_n, size_t max_n)
    {
        min_num = min_n;
        max_num = max_n;
    }

    // 更新网格中的协方差矩阵，并计算其主成分
    void Grid::updateConv()
    {
        // 如果已经更新过，直接返回
        if (is_updated)
            return;
        is_updated = true;

        // 检查点的数量是否足够
        if (points_num < min_num)
        {
            is_valid = false;
            return;
        }

        // 计算协方差
        conv = (conv_sum - points_sum * centroid.transpose()) / (static_cast<double>(points_num) - 1.0);
        is_valid = true;

        // SVD 分解，计算调整后的协方差
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(conv, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d val(1, 1, 1e-3);
        conv = svd.matrixU() * val.asDiagonal() * svd.matrixV().transpose();
        // conv_inv = conv.inverse();
    }

    // 在网格中添加新点
    void Grid::addPoint(Eigen::Vector3d &point, bool insert)
    {
        // 加入新点后标记为未更新
        is_updated = false;

        // 点数
        points_num++;

        // 点坐标的总和
        points_sum += point;

        // 计算点的质心
        // centroid += ((point - centroid) / static_cast<double>(points_num));
        centroid = points_sum / static_cast<double>(points_num);
        conv_sum += (point * point.transpose());

        // 判断是否添加点到列表
        if (points.size() >= max_num * 2 || !insert)
            return;
        points.push_back(point);
    }

    // 根据不同的模式选择邻近的体素
    void VoxelMap::initializeDelta()
    {
        delta_.clear();
        switch (mode_)
        {
        case MODE::NEARBY_1:
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
            break;
        case MODE::NEARBY_7:
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, -resolution_));
            break;
        case MODE::NEARBY_19:
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, -resolution_));
            break;
        default:
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, -resolution_, 0.0));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, 0.0, -resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(0.0, -resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, -resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(resolution_, -resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, resolution_, -resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, -resolution_, resolution_));
            delta_.push_back(Eigen::Vector3d(-resolution_, -resolution_, -resolution_));
            break;
        }
    }

    VoxelMap::VoxelMap(double resolution, size_t capacity, size_t grid_capacity, MODE mode)
        : resolution_(resolution),          // 体素分辨率
          capacity_(capacity),              // 体素容量
          grid_capacity_(grid_capacity),    // 网格容量
          mode_(mode),                      // 邻近模式
          hash_util_(resolution)            // 初始化哈希 hash_util_
    {
        // 存入邻近体素到 delta_
        initializeDelta();
    }

    // 判断是否在同一个网格
    bool VoxelMap::isSameGrid(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
        int hx_1 = floor(p1.x() * resolution_inv_);
        int hy_1 = floor(p1.y() * resolution_inv_);
        int hz_1 = floor(p1.z() * resolution_inv_);
        int hx_2 = floor(p2.x() * resolution_inv_);
        int hy_2 = floor(p2.y() * resolution_inv_);
        int hz_2 = floor(p2.z() * resolution_inv_);

        return ((hx_1 == hx_2) && (hy_1 == hy_2) && (hz_1 == hz_2));
    }

    // 将点云数据加入到体素地图，返回错误计数
    size_t VoxelMap::addCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        size_t error_count = 0;
        if (cloud->empty())
            return 0;

        // 存储当前的体素索引
        current_idx_.clear();

        for (pcl::PointXYZINormal &point : cloud->points)
        {
            // 遍历点云，计算哈希索引，存入 current_idx_
            Eigen::Vector3d point_vec(point.x, point.y, point.z);
            size_t idx = hash_util_(point_vec);
            current_idx_.push_back(idx);

            auto iter = storage_.find(idx);
            if (iter == storage_.end())
            {
                // 如果 storage_ 中不存在，创建一个新的网格
                std::shared_ptr<Grid> grid = std::make_shared<Grid>(grid_capacity_);
                grid->addPoint(point_vec, true);
                cache_.push_front(idx);
                storage_.insert({idx, {cache_.begin(), grid}});

                // 如果超过体素容量，删除最后一个体素
                if (storage_.size() >= capacity_)
                {
                    storage_.erase(cache_.back());
                    cache_.pop_back();
                }
            }
            else
            {
                // 如果存在这个网格，判断点是否在相同的体素网格中
                if (!isSameGrid(iter->second.second->centroid, point_vec))
                {
                    // 如果不在，增加错误计数，创建一个新的体素加入
                    error_count++;
                    cache_.erase(iter->second.first);
                    std::shared_ptr<Grid> grid = std::make_shared<Grid>(grid_capacity_);
                    grid->addPoint(point_vec, true);
                    cache_.push_front(idx);
                    storage_[idx].first = cache_.begin();
                    storage_[idx].second = grid;
                }
                else
                {
                    // 如果在，检查网格中的体素，如果小于 50 则加入点
                    size_t point_in_grid = iter->second.second->points_num;
                    if (point_in_grid < 50)
                        iter->second.second->addPoint(point_vec, point_in_grid < grid_capacity_ ? true : false);
                    cache_.splice(cache_.begin(), cache_, iter->second.first);
                    storage_[idx].first = cache_.begin();
                }
            }
        }

        // 更新新加点的网格的协方差
        for (size_t &idx : current_idx_)
        {
            storage_[idx].second->updateConv();
        }
        return error_count;
    }

    // 重置 cache_ 和 storage_
    void VoxelMap::reset(){
        cache_.clear();
        storage_.clear();
    }

    // 搜索 K 近邻
    bool VoxelMap::searchKNN(const Eigen::Vector3d &point, size_t K, double range, std::vector<Eigen::Vector3d> &results)
    {
        std::vector<std::pair<Eigen::Vector3d, double>> candidates;
        double range2 = range * range;
        for (const Eigen::Vector3d &delta : delta_)
        {
            Eigen::Vector3d near_by = point + delta;
            size_t hash_idx = hash_util_(near_by);
            auto iter = storage_.find(hash_idx);
            if (iter != storage_.end() && isSameGrid(near_by, iter->second.second->centroid))
            {
                for (Eigen::Vector3d &p : iter->second.second->points)
                {
                    double dist = (point - p).squaredNorm();
                    if (dist < range2)
                        candidates.emplace_back(p, dist);
                }
            }
        }

        if (candidates.empty())
            return false;

        if (candidates.size() > K)
        {
            std::nth_element(
                candidates.begin(),
                candidates.begin() + K - 1,
                candidates.end(),
                compare);
            candidates.resize(K);
        }
        std::nth_element(candidates.begin(), candidates.begin(), candidates.end(), compare);

        results.clear();

        for (const auto &it : candidates)
        {
            results.emplace_back(it.first);
        }
        return true;
    }

    // 获取指定体素的质心和协方差矩阵
    bool VoxelMap::getCentroidAndCovariance(size_t hash_idx, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov)
    {
        auto iter = storage_.find(hash_idx);
        if (iter != storage_.end() && iter->second.second->is_valid)
        {
            centroid = iter->second.second->centroid;
            cov = iter->second.second->conv;
            return true;
        }
        return false;
    }

    bool VoxelMap::getCentroidAndCovariance(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov)
    {
        size_t idx = hash_util_(point);
        return getCentroidAndCovariance(idx, centroid, cov);
    }

    void FastGrid::addPoint(const Eigen::Vector3d &point)
    {
        num += 1;
        centroid += ((point - centroid) / static_cast<double>(num));
    }

    FastVoxelMap::FastVoxelMap(double resolution) : resolution_(resolution), hash_util_(resolution)
    {
        // 包含所有相邻体素相对位置的容器 search_range_，用于快速查找体素邻域
        search_range_.clear();
        for (int x_gain = -1; x_gain <= 1; ++x_gain)
        {
            for (int y_gain = -1; y_gain <= 1; ++y_gain)
            {
                for (int z_gain = -1; z_gain <= 1; ++z_gain)
                {
                    search_range_.emplace_back(Eigen::Vector3d(x_gain * resolution_,
                                                               y_gain * resolution_,
                                                               z_gain * resolution_));
                }
            }
        }
    }

    // 处理点云数据，并根据体素网格中心和协方差进行过滤
    void FastVoxelMap::filter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<PointCov> &outs)
    {

        if (cloud->empty())
            return;
        outs.clear();
        grid_array_.clear();
        grid_map_.clear();
        grid_array_.reserve(cloud->size());

        // 遍历点云，插入点到 grid_map_，并且将网格存入 grid_array_
        for (pcl::PointXYZINormal &p : cloud->points)
        {
            Eigen::Vector3d p_vec(p.x, p.y, p.z);
            size_t idx = hash_util_(p_vec);
            auto iter = grid_map_.find(idx);
            if (iter == grid_map_.end())
            {
                std::shared_ptr<FastGrid> grid = std::make_shared<FastGrid>();
                grid->addPoint(p_vec);
                grid_map_.insert({idx, grid});
                grid_array_.push_back(grid);
            }
            else
            {
                iter->second->addPoint(p_vec);
            }
        }

        // 遍历所有网格
        for (std::shared_ptr<FastGrid> &g : grid_array_)
        {

            Eigen::Vector3d points_sum = Eigen::Vector3d::Zero();
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            size_t points_num = 0;

            // 计算相邻体素的中心和协方差之和
            for (Eigen::Vector3d &r : search_range_)
            {
                Eigen::Vector3d near = g->centroid + r;
                auto iter = grid_map_.find(hash_util_(near));
                if (iter != grid_map_.end())
                {
                    points_sum += iter->second->centroid;
                    cov += iter->second->centroid * iter->second->centroid.transpose();
                    points_num += 1;
                }
            }

            // 如果相邻网格大于 6，更新协方差
            if (points_num >= 6)
            {
                Eigen::Vector3d centriod = points_sum / static_cast<double>(points_num);
                cov = (cov - points_sum * centriod.transpose()) / (static_cast<double>(points_num) - 1.0);
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Vector3d values(1, 1, 1e-3);
                cov = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
            }

            // 将网格的中心和调整过的协方差存入 outs
            outs.emplace_back(g->centroid, cov);
        }
    }
} // namespace lio
