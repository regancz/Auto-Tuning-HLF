<template>
  <div class="consortium-container">
    <div class="section">
      <!-- <div class="section-title">联盟</div>
      <el-divider />
      <el-table
        :data="consortiumList"
        style="width: 100%"
        :border="false"
        :stripe="true"
        :header-cell-style="{padding: '10px 20px'}"
        :cell-style="{padding: '10px 20px'}"
      >
        <el-table-column
          prop="name"
          label="名称"
          sortable
          width="220"
          :show-overflow-tooltip="true"
        />
        <el-table-column
          prop="domain"
          label="域名"
          sortable
          width="280"
          :show-overflow-tooltip="true"
        />
        <el-table-column
          prop="peerCount"
          label="通道数目"
          sortable
          width="220"
          :show-overflow-tooltip="true"
        />
        <el-table-column
          prop="status"
          label="创建日期"
          sortable
          width="250"
          :show-overflow-tooltip="true"
        />
        <el-table-column
          prop="status"
          label="状态"
          sortable
          width="250"
          :show-overflow-tooltip="true"
        />
        <el-table-column
          prop="remarks"
          label="备注"
          width="200"
          :formatter="formatter"
          :show-overflow-tooltip="true"
        />
      </el-table>
    </div> -->

      <div class="section">
        <div class="section-title">通道</div>
        <el-divider />
        <el-table
          :data="channelList"
          style="width: 100%"
          :border="false"
          :stripe="true"
          :header-cell-style="{padding: '10px 20px'}"
          :cell-style="{padding: '10px 20px'}"
        >
          <el-table-column
            prop="name"
            label="名称"
            sortable
            width="220"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="domain"
            label="域名"
            sortable
            width="280"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="orgCount"
            label="组织数目"
            sortable
            width="200"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="ordererCount"
            label="Orderer数目"
            sortable
            width="200"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="version"
            label="版本"
            sortable
            width="150"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="createTime"
            label="创建时间"
            sortable
            width="220"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="status"
            label="状态"
            sortable
            width="150"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="remarks"
            label="备注"
            width="200"
            :formatter="formatter"
            :show-overflow-tooltip="true"
          />
        </el-table>
      </div>

      <div class="section">
        <div class="section-title">组织</div>
        <el-divider />
        <el-table
          :data="orgList"
          style="width: 100%"
          :border="false"
          :stripe="true"
          :header-cell-style="{padding: '10px 20px'}"
          :cell-style="{padding: '10px 20px'}"
        >
          <el-table-column
            prop="name"
            label="名称"
            sortable
            width="220"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="domain"
            label="域名"
            sortable
            width="280"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="peerCount"
            label="Peer数目"
            sortable
            width="220"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="version"
            label="版本"
            sortable
            width="250"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="status"
            label="状态"
            sortable
            width="250"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="remarks"
            label="备注"
            width="200"
            :formatter="formatter"
            :show-overflow-tooltip="true"
          />
        </el-table>
      </div>
      <div class="section">
        <div class="section-title">链码</div>
        <el-divider />
        <el-table
          :data="chaincodeList"
          style="width: 100%"
          :border="false"
          :stripe="true"
          :header-cell-style="{padding: '10px 20px'}"
          :cell-style="{padding: '10px 20px'}"
        >
          <el-table-column
            prop="name"
            label="名称"
            sortable
            width="220"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="domain"
            label="域名"
            sortable
            width="280"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="version"
            label="版本"
            sortable
            width="250"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="status"
            label="状态"
            sortable
            width="250"
            :show-overflow-tooltip="true"
          />
          <el-table-column
            prop="remarks"
            label="备注"
            width="200"
            :show-overflow-tooltip="true"
          />
        </el-table>
      </div>
    </div>

  </div></template>

<script>
import { queryAllChaincode, queryAllChannel, queryAllOrg } from '@/api/overview'
export default {
  data() {
    return {
      channelList: [],
      orgList: [],
      chaincodeList: []
    }
  },
  mounted() {
    this.queryChannelData()
    this.queryOrgData()
    this.queryChaincodeData()
  },
  methods: {
    async queryChannelData() {
      try {
        // Call your API to query channel data
        const response = await queryAllChannel()
        this.channelList = response.data // Assign the queried data to channelList
      } catch (error) {
        console.error('Error querying channel data:', error)
      }
    },
    async queryOrgData() {
      try {
        // Call your API to query org data
        const response = await queryAllOrg()
        this.orgList = response.data // Assign the queried data to orgList
      } catch (error) {
        console.error('Error querying org data:', error)
      }
    },
    async queryChaincodeData() {
      try {
        // Call your API to query chaincode data
        const response = await queryAllChaincode()
        this.chaincodeList = response.data // Assign the queried data to chaincodeList
      } catch (error) {
        console.error('Error querying chaincode data:', error)
      }
    }
  }
}
</script>

<style scoped>
.section {
  flex: 1;
  overflow: auto;
}

.section-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 10px;
}

.consortium-container {
  margin: 20px;
}

.consortium-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 10px;
}

.separator {
  margin: 10px 0;
  border: 1px solid #ccc;
}

.consortium-list {
  display: flex;
  flex-wrap: wrap;
}

.consortium-item {
  width: 100%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 8px;
  margin: 10px 0;
  box-sizing: border-box;
}

.consortium-info {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
}

.info-label {
  font-weight: bold;
  margin-right: 5px;
}

.info-value {
  flex-grow: 1;
}

.el-table {
  margin-bottom: 20px;
}
</style>
