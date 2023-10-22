<template>
    <div id="chatRuleBox-root">
        <p>{{contactText}}</p>
        <el-link type="primary" @click="showRuleTips">Show related entities</el-link>
    </div>
</template>

<script>
export default {
  name: "chatRuleBox",
  props:['contactText'],
  data() {
    return {
      
    };
  },
  components: {
  },
  methods: {
    showRuleTips(){
        this.$emit('isShowRuleTips',true);
    },
  },
};
</script>

<style scoped lang="less">
#chatRuleBox-root{
}
</style>