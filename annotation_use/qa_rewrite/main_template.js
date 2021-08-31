$(document).ready(function() {

    //localStorage.clear();
    
    var version = 1
    var index = 0
    var qa_info_list = []

    //annotation set
    // <replace_item_cat>
    // <replace_item_prod>
    // <replace_item_ques>
    // <replace_item_ans>

    //annotation set
    var prod_cat_list =['Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products', 'Office_Products']
    var prod_idx_list =['B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B006WAPTVM', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B00BLVLHZE', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO', 'B0047G19YO']
    var ques_list =['How large of an image can be scanned?', "If I''m not predominantly printing photographs, should I still buy this? I mainly need it for printing illustrations, and other designed materials.", 'Can this printer be hard-wired (USB cable) to the computer instead of Wi-Fi?', 'I am a visual artist who typically works in mixed media. Would this be a good printer for me? I am interested in archival, matte, giclee prints.', "Does it bother anyone that it doesn't have a scanner? (Do you just keep a separate scanner?)", 'How is this printer at regular text printing?', 'I need to make sure this printer will FEED and print VERY HEAVY cardstock. I *think* the card stock is 110 lb. Anyone have experience here?', 'what is the maintenance on the printer', "Did anyone else have the problem that says NO LINE even when it's a working phone line?", 'does the base display show the phone number or contact of an incoming call?  I like being able to glance over at the phone and see who is calling.', "I'm confused. Is it a landline as well as a connect to cell? I just got a landline number.", 'Can this device be used WITHOUT a landline and ONLY a CELL phone?', 'Can the hand set be used as a speaker phone. And if yes, how is the sound quality? Thanks', 'Does this work with a Samsung Note 3?', 'If you have an incoming call, how do you answer?  Can you just pick up the handset to answer, and hang it up to end; or do you have to push a button?', 'Does this unit have a 2 landline capability? I need a phone with 2 landlines. Thank You for your answer', 'Would you buy this chair again? Is it still comfortable?', 'how much weight will this chair hold?', 'What is the weight capacity?', 'Does this chair leans back?', 'i want to know how low will the chair go down to', 'How high will the seat adjust to? I need at least 24-25 inches because my computer table is pretty high.', "I just put this chair together. Uhh. Any ideas on why it will not deflate down to its original size? I brought it up and now it won't go back down", 'What is the width of the wheel base? I have a narrow desk that can accommodate 21inches across']
    var ans_list =['Since the 1430 is not a scanner I assume you want to know how large an image can be printed.  The width limit is 13 inches, but lengthwise it will print panoramas that are quite long.  I have some prints (using rolled paper) that are 44 inches long (i.e.13" x 44").  Most of my large prints are typically 11 x 17 or 13 x 19, or in that general range (depending on crop) as these are more common paper sizes.Hope this helps.', "We don't really use it for photos. We use it because it accommodates a wide range of paper and mostly use it for graphic design, like cards and invites.", 'Yes by default. The setup disc provides the option', 'I use this printer for printing my photographs. Epson has a wide range of papers. Their Signature Worthy brand offers a variety of finishes. These papers are expensive, but I find that for my photos they print very well. The prints at 13x19 are amazing.', 'I have a separate scanner', 'I never use it for anything other than for color prints...', "I've used heavy weight photo and card stock successfully.", 'I have had no maintenance on mine, love it.  Janet R', 'no. I hadn´t.', 'It does -  but is isnot the easiest to see.', 'Yes, it is both. You connect it to your landline, and sync it with your Bluetooth cell. You can answer either line from this phone.', 'Yes. We only have the cell phone connected via Bluetooth. It works really well too.', 'Yes, the hand set can be used as a speaker phone. The sound quality is average. The hand set is very light weight.', "Much better than we expected. We can get a cell signal only in a couple window sills of this house. We tried an expensive signal booster, but it didn't help. The signal kept dropping. My husband put this device on his office window sill in a far corner of the basement. He remains connected and clear on his cell phone throughout the basement and through most of the main floor, and this is a pretty big house. We bought it quite a while ago, but we've used it only for about a month. So far, so good.", "You can either pick up a handset and hit the 'home' button, or you can just push a button on the main unit (the 'home' button) and it will answer as a speakerphone.", 'No, only has one landline.', 'I would buy this chair again.  It comfortably serves as my kitchen desk chair.  It is used daily by all members of the family.', "I don't actually know the answer to the question (did the manufacturer post anything?) but both my son and I are comfortable and feel safe in this chair -- no problem.  (130, 120 lbs.)  We still love it!", 'It was either 200 or 250 lbs. Should be on seller info. If anyone was over 250 lbs., I\'d research under "bariatric chairs." I think bariatric chairs go up to 500-ish lbs.', "Nope. The back is fixed to the seat with a metal bar. There's no flexibility there. The seat doesn't tilt on the base, either.", '14 inches from floor to top of seat part and 19 inches when up all the way.', 'I am having the same problem finding a chair high enough. Did you ever have any luck finding a chair?', "Nope no idea.  It's a cheap chair.", 'It\'s approximately 23" across from wheel to wheel.']

    
    var len = prod_idx_list.length

    // check porgress and restore
    //qa_info_list = JSON.parse(localStorage.getItem("data_info"));  
    //index = parseInt(localStorage.getItem("progress_index"));
    

    //if (qa_info_list == null){ qa_info_list = [] }
    //if (isNaN(index)){ index = 0 }

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

    $('#submit_panel').hide(0).delay(150000).show(0);  

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

        $('#submit_panel').hide(0).delay(150000).show(0);
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
        //localStorage.setItem("data_info", JSON.stringify(qa_info_list));
        //localStorage.setItem("progress_index", JSON.stringify(index));

        if (index == len) { 
            $('#question_form').hide();
            $("#answer").val(JSON.stringify(qa_info_list));
            $("#mturk_form").submit();
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
