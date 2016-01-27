$(document).ready( function () {
  $(".expandtbl1").hide();
} );

$(document).ready(function(){
  // $(".expandtbl1").hide();
  $(".btntbl").click(function(){
    $(".expandtbl1").toggle();
  });
  $(".btntbl2").click(function(){
    $(".expandtbl2").toggle();
  });
  $('#table_id').DataTable();
  $(window).height();
});
