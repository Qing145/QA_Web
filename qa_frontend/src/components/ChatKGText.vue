<template>
    <div id="chatKGText-root">
        <p>{{contactText}}</p>
        <el-link type="primary" @click="showKG">Knowledge Graph</el-link>
    </div>
</template>

<script>
export default {
  name: "chatKGText",
  props:['contactText'],
  data() {
    return {
      
    };
  },
  components: {
  },
  methods: {
    showKG(){
        this.$emit('isShowKG',true);
    },
  },
};
</script>

<style scoped lang="less">
#chatKGText-root{
}
</style>