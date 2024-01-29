<template>
  <div class="log-container">
    <el-form :inline="true" :model="formInline" class="demo-form-inline">
      <el-form-item label="ID">
        <el-input v-model="formInline.id" placeholder="ID" />
      </el-form-item>
      <el-form-item label="执行器">
        <el-select v-model="formInline.taskType" placeholder="请输入执行器">
          <el-option label="Collector" value="Collector" />
          <el-option label="Controller" value="Controller" />
          <el-option label="Analyzer" value="Analyzer" />
          <el-option label="Optimizer" value="Optimizer" />
        </el-select>
      </el-form-item>
      <el-form-item label="时间">
        <el-input v-model="formInline.timestamp" placeholder="请输入时间" />
      </el-form-item>
      <el-form-item label="日志级别">
        <el-select v-model="formInline.level" placeholder="请输入日志级别">
          <el-option label="INFO" value="CollecINFOtor" />
          <el-option label="ERROR" value="ERROR" />
          <el-option label="WARN" value="WARN" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">查询</el-button>
      </el-form-item>
    </el-form>
    <el-card class="log-card">
      <el-table :data="pagedLogs" style="width: 100%" border :default-sort="{ prop: 'timestamp', order: 'descending' }">
        <el-table-column prop="id" label="ID" width="350" sortable />
        <el-table-column prop="taskType" label="执行器" width="120" sortable />
        <el-table-column prop="timestamp" label="时间" width="240" sortable />
        <el-table-column prop="level" label="级别" width="100" sortable />
        <el-table-column prop="message" label="消息" sortable />
      </el-table>
      <el-pagination
        background
        layout="prev, pager, next"
        :total="totalLogs"
        :page-size="pageSize"
        @current-change="handleCurrentChange"
        @size-change="handleSizeChange"
      />
    </el-card>
  </div>
</template>

<script>
import { queryAllLog } from '@/api/log'
export default {
  data() {
    return {
      logs: [],
      pagedLogs: [],
      totalLogs: 0,
      pageSize: 12,
      currentPage: 1,
      formInline: {
        id: '',
        taskType: '',
        timestamp: '',
        level: ''
      }
    }
  },
  mounted() {
    this.queryAllLog()
  },
  methods: {
    onSubmit() {
      console.log('submit!')
    },
    async queryAllLog() {
      try {
        const response = await queryAllLog() // Assuming queryAllLog returns a Promise
        this.logs = response.data // Assign queried data to logs
        this.totalLogs = this.logs.length
        this.updatePagedLogs()
      } catch (error) {
        console.error('Error querying log data:', error)
      }
    },
    handleCurrentChange(newPage) {
      console.info(newPage)
      this.currentPage = newPage
      this.updatePagedLogs()
    },
    updatePagedLogs() {
      this.pagedLogs = this.logs.slice((this.currentPage - 1) * this.pageSize, this.currentPage * this.pageSize)
    },
    handleSizeChange(size) {
      console.info(size)
      this.pagesize = size
      this.updatePagedLogs()
    }
  }
}
</script>

<style scoped>
.log-container {
  padding: 20px;
}

.log-card {
  margin-top: 20px;
}

.log-card .el-button {
  margin-bottom: 10px;
}
</style>
