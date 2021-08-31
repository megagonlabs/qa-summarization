$(document).ready(function() {

    localStorage.clear();
    
    var version = 1
    var index = 0
    var qa_info_list = []

    //annotation set
    <replace_item_cat>
    <replace_item_prod>
    <replace_item_ques>
    <replace_item_ans>
    
    var len = prod_idx_list.length

    // check porgress and restore
    qa_info_list = JSON.parse(localStorage.getItem("data_info"));  
    index = parseInt(localStorage.getItem("progress_index"));
    

    if (qa_info_list == null){ qa_info_list = [] }
    if (isNaN(index)){ index = 0 }

    // default data info
    for (i = 0; i < 8; i++){
        var q_txt = '.ques_txt_' + parseInt(i+1) + ' span'
        var a_txt = '.ans_txt_' + parseInt(i+1) + ' span'
        $(q_txt).text(ques_list[i])
        $(a_txt).text(ans_list[i])
    }

    $('.prod_cat span').text(prod_cat_list[index])
    $('.prod_idx span').text(prod_idx_list[index])

    var url = "https://www.amazon.com/dp/" + prod_idx_list[index]
    $("a").prop("href", url)
    
    $('input[type="radio"]').prop('checked', false);
    
    $(document).on('click', 'a', function(e){ 
        e.preventDefault(); 
        var url = $(this).attr('href'); 
        window.open(url, '_blank');
    });

    $('#submit_panel').hide(0).delay(1000).show(0);  

    $('#error_example').hide();
    $('.error_btn').click(function(){

        if ($('.error_btn').text() == 'show examples'){
            $('.error_btn').text('hide examples');
            $('#error_example').show();
        }
        else if ($('.error_btn').text() == 'hide examples'){
            $('.error_btn').text('show examples');
            $('#error_example').hide();
        }
    }); 

    $(document).on('click', '#submit_btn', function(evt) {
        
        var qa_pair_list = [];
        for (i = 0; i < 8; i++){
            var check = "input[name='qa"+parseInt(i+1)+"']:checked"
            $.each($(check), function(){ 
            qa_pair_list.push($(this).val());         
            });
        }

        setTimeout(function(){
        $("#warning").hide();
        },2000)

        if (qa_pair_list.length < 8) {
            $("#warning").show();
            $("#warning").text("Please finish the question!");
            return;
        } 

        // $('#submit_panel').hide(0).delay(20000).show(0);
        var ques_txt = []
        var ans_txt = []
        for (i = index; i < index+8; i++){
            ques_txt.push(ques_list[i]);
            ans_txt.push(ans_list[i]);         
        }

        index += 8
        
        // save data information before next figure
        var prod_cat = $('.prod_cat span').text()
        var prod_idx = $('.prod_idx span').text()
        var data_info = {'prod_cat':prod_cat, 'prod_idx':prod_idx, 'qa_quality':qa_pair_list,
                        'ques':ques_txt, 'ans':ans_txt}

        qa_info_list.push(data_info)
        localStorage.setItem("data_info", JSON.stringify(qa_info_list));
        localStorage.setItem("progress_index", JSON.stringify(index));

        if (index == len) { 
            $('#question_form').hide();
            $("#answer").val(JSON.stringify(qa_info_list));
            $("#mturk_form").submit();
            console.log(JSON.stringify(qa_info_list))
        }

        // update data info
        $('.prod_cat span').text(prod_cat_list[index])
        $('.prod_idx span').text(prod_idx_list[index])
        for (i = index; i < index+8; i++){
            var q_txt = '.ques_txt_' + parseInt(i%8+1) + ' span'
            var a_txt = '.ans_txt_' + parseInt(i%8+1) + ' span'
            $(q_txt).text(ques_list[i])
            $(a_txt).text(ans_list[i])
        }

        $('input[type="radio"]').prop('checked', false);

        $('.progress-bar').attr('aria-valuenow',index);
        $('.progress-bar').attr('style','width:'+(index/len*100).toFixed(2)+'%');
        $('.progress-bar').text((index/len*100).toFixed(2)+'%');
        $('#prog_count').text((index/len*100).toFixed(2)+'% '+'complete');

        
        var url = "https://www.amazon.com/dp/" + prod_idx_list[index]
        $("a").prop("href", url)
    });
});

var temp;