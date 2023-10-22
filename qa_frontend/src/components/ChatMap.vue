<template>
  <div id="chatMap-root">
    <p v-for="(item, index) in contactText.answer" :key="index">
      {{ item[0] }}: {{ item[1] }}
    </p>
    <el-link type="primary" @click="showMap">Show Map</el-link>
  </div>
</template>

<script>
export default {
  name: "chatMap",
  props: ["contactText"],
  data() {
    return {};
  },
  components: {},
  methods: {
    showMap() {
      let that = this;
      this.$emit("isShowMap", that.contactText.coordinates);
    },
  },
};
</script>

<style scoped lang="less">
</style>