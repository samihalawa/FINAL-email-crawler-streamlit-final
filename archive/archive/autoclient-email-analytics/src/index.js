import { createClient } from "@supabase/supabase-js";
import renderHtml from "./renderHtml.js";

const RETRY_ATTEMPTS = 3;
const RETRY_DELAY = 1000; // 1 second

async function retryOperation(operation, attempts = RETRY_ATTEMPTS) {
  for (let i = 0; i < attempts; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === attempts - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
    }
  }
}

function calculateAverageTimeMetric(campaigns, endTimeField, startTimeField) {
  const validCampaigns = campaigns.filter(c => c[endTimeField] && c[startTimeField]);
  if (validCampaigns.length === 0) return 0;
  
  const totalTime = validCampaigns.reduce((sum, campaign) => {
    const endTime = new Date(campaign[endTimeField]);
    const startTime = new Date(campaign[startTimeField]);
    return sum + (endTime - startTime);
  }, 0);
  
  return Math.round(totalTime / validCampaigns.length / 1000 / 60); // Returns minutes
}

async function updateAnalytics(env, trackingId, trackingType) {
  try {
    const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_KEY);
    
    // First get the current campaign data
    const { data: campaign, error: fetchError } = await supabase
      .from('email_campaigns')
      .select('id, open_count, click_count, status')
      .eq('tracking_id', trackingId)
      .single();

    if (fetchError || !campaign) {
      console.error('Error fetching campaign:', fetchError);
      return;
    }

    // Prepare update data
    const updateData = {
      status: campaign.status === 'sent' ? 'engaged' : campaign.status
    };

    if (trackingType === 'open') {
      updateData.opened_at = new Date().toISOString();
      updateData.open_count = (campaign.open_count || 0) + 1;
    } else if (trackingType === 'click') {
      updateData.clicked_at = new Date().toISOString();
      updateData.click_count = (campaign.click_count || 0) + 1;
    }

    // Update the campaign
    const { error: updateError } = await supabase
      .from('email_campaigns')
      .update(updateData)
      .eq('id', campaign.id);

    if (updateError) {
      console.error('Error updating campaign:', updateError);
    }
  } catch (error) {
    console.error('Analytics update error:', error);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check endpoint
    if (url.pathname === '/health') {
      return new Response('OK', { 
        status: 200,
        headers: {
          ...corsHeaders,
          'Cache-Control': 'no-store'
        }
      });
    }

    const trackingId = url.searchParams.get('id');
    const trackingType = url.searchParams.get('type');
    const originalUrl = url.searchParams.get('url');

    if (trackingId && trackingType) {
      // Fire and forget analytics update
      updateAnalytics(env, trackingId, trackingType)
        .catch(error => console.error('Background analytics update failed:', error));

      if (trackingType === 'open') {
        // Return 1x1 transparent GIF immediately
        return new Response(
          new Uint8Array([71, 73, 70, 56, 57, 97, 1, 0, 1, 0, 128, 0, 0, 255, 255, 255, 0, 0, 0, 33, 249, 4, 1, 0, 0, 0, 0, 44, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 68, 1, 0, 59]),
          { 
            headers: { 
              ...corsHeaders,
              'Content-Type': 'image/gif',
              'Cache-Control': 'no-store, no-cache, must-revalidate',
              'Pragma': 'no-cache'
            } 
          }
        );
      } else if (trackingType === 'click' && originalUrl) {
        // Redirect immediately
        return Response.redirect(originalUrl, 302);
      }
    }

    // Show dashboard for non-tracking requests
    try {
      const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_KEY);
      const [campaignStats, automationLogs] = await Promise.all([
        retryOperation(async () => {
          const { data, error } = await supabase
            .from('email_campaigns')
            .select(`
              id, 
              sent_at, 
              status, 
              open_count, 
              click_count,
              opened_at,
              clicked_at,
              lead_id
            `)
            .order('sent_at', { ascending: false })
            .limit(1000);

          if (error) throw error;
          return data;
        }),
        retryOperation(async () => {
          const { data, error } = await supabase
            .from('automation_logs')
            .select('logs, start_time')
            .order('start_time', { ascending: false })
            .limit(20);

          if (error) throw error;
          return data;
        })
      ]);

      const totalSent = campaignStats.length;
      const totalOpened = campaignStats.filter(ec => ec.open_count > 0).length;
      const totalClicked = campaignStats.filter(ec => ec.click_count > 0).length;
      const openRate = totalSent > 0 ? (totalOpened / totalSent * 100).toFixed(2) : 0;
      const clickRate = totalSent > 0 ? (totalClicked / totalSent * 100).toFixed(2) : 0;
      const averageTimeToOpen = calculateAverageTimeMetric(campaignStats, 'opened_at', 'sent_at');
      const averageTimeToClick = calculateAverageTimeMetric(campaignStats, 'clicked_at', 'sent_at');

      const logs = automationLogs.flatMap(log => 
        Array.isArray(log.logs) 
          ? log.logs.map(l => `${new Date(log.start_time).toLocaleString()}: ${l}`)
          : [`${new Date(log.start_time).toLocaleString()}: ${log.logs}`]
      );

      const html = await renderHtml(
        { 
          totalSent, 
          totalOpened, 
          totalClicked, 
          openRate, 
          clickRate,
          averageTimeToOpen,
          averageTimeToClick
        },
        logs
      );

      return new Response(html, {
        headers: { 
          ...corsHeaders,
          'Content-Type': 'text/html',
          'Cache-Control': 'no-cache'
        },
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      return new Response('Internal Server Error', { 
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'text/plain',
          'Cache-Control': 'no-store'
        }
      });
    }
  }
};