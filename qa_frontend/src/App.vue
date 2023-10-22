<template>
  <div id="app">
    <div id="content">
      <transition :name="transitionName">
        <router-view />
      </transition>
    </div>
  </div>
</template>
  
<script>
export default {
  name: "app",
  data() {
    return {
      transitionName: "",
    };
  },
  watch: {
    $route(to, from) {
      //实现路由跳转动画
      if (to.meta.index > from.meta.index) this.transitionName = "slide-left";
      if (to.meta.index < from.meta.index) this.transitionName = "slide-right";
    },
  },
};
</script>
  
<style lang="less">
html,
body {
  margin: 0px;
  padding: 0px;
  height: 100%;
  // width: 100%;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #000;
  overflow: hidden;
}

#content {
  display: inline-block;
  width: 80%;
  height: 80%;
  min-width: 900px;
  position: relative;
}

.slide-right-enter-active,
.slide-right-leave-active,
.slide-left-enter-active,
.slide-left-leave-active {
  will-change: transform;
  transition: all 500ms;
  position: absolute !important;
}

.slide-right-enter {
  opacity: 0;
  transform: translate(-100%);
}

.slide-right-leave-active {
  opacity: 0;
  transform: translateX(100%);
}

.slide-left-enter {
  opacity: 0;
  transform: translateX(100%);
}

.slide-left-leave-active {
  opacity: 0;
  transform: translateX(-100%);
}
</style>
  