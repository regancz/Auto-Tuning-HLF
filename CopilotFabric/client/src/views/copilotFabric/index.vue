<template>
  <div class="dashboard-container">
    <el-row :gutter="40" class="panel-group">
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-people">
            <svg-icon icon-class="customer-group" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Peer Nodes
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="peerNodesCount" :duration="2600" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-message">
            <svg-icon icon-class="customer-group" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Organizations
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="organizations" :duration="3000" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-money">
            <svg-icon icon-class="order" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Orderer Nodes
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="ordererNodesCount" :duration="3200" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-shopping">
            <svg-icon icon-class="ca" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              CA Server
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="caCount" :duration="3600" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-people">
            <svg-icon icon-class="channel" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Channel
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="channel" :duration="2600" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-message">
            <svg-icon icon-class="block" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Block Height
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="blockHeight" :duration="3000" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-money">
            <svg-icon icon-class="transaction" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Transaction
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="transactionCount" :duration="3200" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-shopping">
            <svg-icon icon-class="chaincode" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              Chaincode
            </div>
            <div style="text-align: right;">
              <count-to :start-val="0" :end-val="chaincodeCount" :duration="3600" class="card-panel-num" />
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import CountTo from 'vue-count-to'
import { getBlockHeight, getCACount, getChaincodeCount, getChannel, getOrdererNodesCount, getOrganizations, getPeerNodesCount, getTransactionCount } from '@/api/dashboard'
import { Message } from 'element-ui'

export default {
  components: {
    CountTo
  },
  data() {
    return {
      blockHeight: 0,
      caCount: 0,
      chaincodeCount: 0,
      channel: 0,
      ordererNodesCount: 0,
      organizations: 0,
      peerNodesCount: 0,
      transactionCount: 0
    }
  },
  mounted() {
    this.fetchDataAndUpdate()
  },
  methods: {
    fetchDataAndUpdate() {
    // 封装所有异步请求
      const requests = [
        getBlockHeight(),
        getCACount(),
        getChaincodeCount(),
        getChannel(),
        getOrdererNodesCount(),
        getOrganizations(),
        getPeerNodesCount(),
        getTransactionCount()
      ]

      // 执行所有请求并处理响应
      Promise.all(requests.map(request => request.catch(error => ({ error }))))
        .then(responses => {
        // 处理每个请求的响应
          this.blockHeight = this.extractData(responses[0])
          this.caCount = this.extractData(responses[1])
          this.chaincodeCount = this.extractData(responses[2])
          this.channel = this.extractData(responses[3])
          this.ordererNodesCount = this.extractData(responses[4])
          this.organizations = this.extractData(responses[5])
          this.peerNodesCount = this.extractData(responses[6])
          this.transactionCount = this.extractData(responses[7])
        })
        .catch(error => {
          console.error(error) // 打印详细错误信息
          console.log('err' + error) // for debug
          Message({
            message: error.message,
            type: 'error',
            duration: 5 * 1000
          })
          return Promise.reject(error)
        })
    },
    extractData(response) {
      return response && !response.error ? response.data.count : undefined
    },
    handleSetLineChartData(type) {
      this.$emit('handleSetLineChartData', type)
    }
  }
}
</script>

<style lang="scss" scoped>
.panel-group {
  margin-top: 18px;
  padding: 0 10px;

  .card-panel-col {
    margin-bottom: 32px;
  }

  .card-panel {
    height: 108px;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
    border-color: rgba(0, 0, 0, .05);

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-people {
        background: #40c9c6;
      }

      .icon-message {
        background: #36a3f7;
      }

      .icon-money {
        background: #f4516c;
      }

      .icon-shopping {
        background: #34bfa3
      }
    }

    .icon-people {
      color: #40c9c6;
    }

    .icon-message {
      color: #36a3f7;
    }

    .icon-money {
      color: #f4516c;
    }

    .icon-shopping {
      color: #34bfa3
    }

    .card-panel-icon-wrapper {
      float: left;
      margin: 14px 0 0 14px;
      padding: 16px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-icon {
      float: left;
      font-size: 48px;
    }

    .card-panel-description {
      float: right;
      font-weight: bold;
      margin: 26px;
      margin-left: 0px;

      .card-panel-text {
        line-height: 18px;
        color: rgba(0, 0, 0, 0.45);
        font-size: 16px;
        margin-bottom: 12px;
      }

      .card-panel-num {
        font-size: 20px;
      }
    }
  }
}

@media (max-width:550px) {
  .card-panel-description {
    display: none;
  }

  .card-panel-icon-wrapper {
    float: none !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;

    .svg-icon {
      display: block;
      margin: 14px auto !important;
      float: none !important;
    }
  }
}
</style>
