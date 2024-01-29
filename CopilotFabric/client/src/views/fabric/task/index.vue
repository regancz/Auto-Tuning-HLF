<template>
  <div class="scheduler-container">
    <el-form :inline="true" :model="taskForm" class="demo-form-inline">
      <el-form-item label="任务类型">
        <el-select v-model="taskForm.taskType" placeholder="请选择任务类型">
          <el-option label="Cron" value="Cron" />
          <el-option label="Fixed rate" value="Fixed rate" />
          <el-option label="One Time" value="One Time" />
        </el-select>
      </el-form-item>
      <el-form-item v-if="taskForm.taskType === 'Cron'" label="定时规则" prop="cronExpression">
        <el-input v-model="taskForm.cronExpression" placeholder="请输入定时规则" />
      </el-form-item>
      <el-form-item v-if="taskForm.taskType === 'Fixed rate'" label="Rate" prop="fixedRate">
        <el-input v-model="taskForm.fixedRate" placeholder="请输入Rate参数" />
      </el-form-item>
      <el-form-item label="算法" prop="algorithm">
        <el-select v-model="taskForm.algorithm" placeholder="请选择算法">
          <el-option label="SPSA" value="SPSA" />
          <el-option label="ASPSA" value="ASPSA" />
          <el-option label="MOASPSA" value="MOASPSA" />
          <el-option label="MOPSO" value="MOPSO" />
          <el-option label="BPNN" value="BPNN" />
          <el-option label="SVR" value="SVR" />
        </el-select>
      </el-form-item>
      <el-form-item label="通道" prop="channel">
        <el-input v-model="taskForm.channel" placeholder="请选择通道" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="submitTaskForm">发布任务</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="taskList" style="margin-top: 20px; flex: 1;">
      <el-table-column label="ID" prop="uuid" width="400" />
      <el-table-column label="任务类型" prop="taskType" />
      <el-table-column label="参数" prop="cronExpression" />
      <el-table-column label="算法" prop="algorithm" />
      <el-table-column label="通道" prop="channel" />
      <el-table-column label="状态" prop="status" />
    </el-table>
  </div>
</template>

<script>
export default {
  data() {
    return {
      taskForm: {
        uuid: '', // 自动生成
        taskType: '',
        cronExpression: '',
        fixedRate: '',
        algorithm: '',
        channel: ''
      },
      taskList: []
    }
  },
  methods: {
    submitTaskForm() {
      // 模拟发布任务，将任务配置加入任务列表
      this.taskList.push({
        uuid: this.generateUUID(),
        taskType: this.taskForm.taskType,
        cronExpression: this.taskForm.cronExpression,
        fixedRate: this.taskForm.fixedRate,
        algorithm: this.taskForm.algorithm,
        channel: this.taskForm.channel,
        status: '待执行'
      })

      // 清空表单
      this.$refs.taskForm.resetFields()
    },
    generateUUID() {
      // 简单的UUID生成函数，可以根据实际情况进行改进
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (Math.random() * 16) | 0
        var v = c === 'x' ? r : (r & 0x3) | 0x8
        return v.toString(16)
      })
    }
  }
}
</script>

<style scoped>
.scheduler-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 20px;
}

.el-form-item__label {
  font-weight: bold;
}

.el-input,
.el-button,
.el-select {
  width: 100%;
  margin-top: 10px;
}

.el-table {
  margin-top: 20px;
  flex: 1;
}
</style>
